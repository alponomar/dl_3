       �K"	   Ku�Abrain.Event:2��|ne�     �4��	wHKu�A"ؤ
[
xPlaceholder*
dtype0*
shape: */
_output_shapes
:���������  
S
yPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

p
train_cnn/Reshape/shapeConst*
dtype0*%
valueB"����           *
_output_shapes
:
�
train_cnn/ReshapeReshapextrain_cnn/Reshape/shape*/
_output_shapes
:���������  *
T0*
Tshape0
�
ConvNet/conv1/WVariable*
dtype0*
shape:@*
	container *
shared_name *&
_output_shapes
:@
�
/ConvNet/conv1/W/Initializer/random_normal/shapeConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*%
valueB"         @   *
_output_shapes
:
�
.ConvNet/conv1/W/Initializer/random_normal/meanConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *    *
_output_shapes
: 
�
0ConvNet/conv1/W/Initializer/random_normal/stddevConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *o�:*
_output_shapes
: 
�
>ConvNet/conv1/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/ConvNet/conv1/W/Initializer/random_normal/shape*&
_output_shapes
:@*
dtype0*
seed2*

seed**
T0*"
_class
loc:@ConvNet/conv1/W
�
-ConvNet/conv1/W/Initializer/random_normal/mulMul>ConvNet/conv1/W/Initializer/random_normal/RandomStandardNormal0ConvNet/conv1/W/Initializer/random_normal/stddev*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
�
)ConvNet/conv1/W/Initializer/random_normalAdd-ConvNet/conv1/W/Initializer/random_normal/mul.ConvNet/conv1/W/Initializer/random_normal/mean*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
�
ConvNet/conv1/W/AssignAssignConvNet/conv1/W)ConvNet/conv1/W/Initializer/random_normal*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
�
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
�
!ConvNet/conv1/b/Initializer/ConstConst*
dtype0*"
_class
loc:@ConvNet/conv1/b*
valueB@*    *
_output_shapes
:@
�
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
�
train_cnn/ConvNet/conv1/Conv2DConv2Dtrain_cnn/ReshapeConvNet/conv1/W/read*/
_output_shapes
:���������  @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
train_cnn/ConvNet/conv1/addAddtrain_cnn/ConvNet/conv1/Conv2DConvNet/conv1/b/read*
T0*/
_output_shapes
:���������  @
{
train_cnn/ConvNet/conv1/ReluRelutrain_cnn/ConvNet/conv1/add*
T0*/
_output_shapes
:���������  @
�
train_cnn/ConvNet/conv1/MaxPoolMaxPooltrain_cnn/ConvNet/conv1/Relu*/
_output_shapes
:���������@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
z
,train_cnn/ConvNet/conv1/HistogramSummary/tagConst*
dtype0*
valueB Bconv1_weights*
_output_shapes
: 
�
(train_cnn/ConvNet/conv1/HistogramSummaryHistogramSummary,train_cnn/ConvNet/conv1/HistogramSummary/tagConvNet/conv1/W/read*
T0*
_output_shapes
: 
v
.train_cnn/ConvNet/conv1/HistogramSummary_1/tagConst*
dtype0*
valueB Bconv1_b*
_output_shapes
: 
�
*train_cnn/ConvNet/conv1/HistogramSummary_1HistogramSummary.train_cnn/ConvNet/conv1/HistogramSummary_1/tagConvNet/conv1/b/read*
T0*
_output_shapes
: 
x
.train_cnn/ConvNet/conv1/HistogramSummary_2/tagConst*
dtype0*
valueB B	conv1_out*
_output_shapes
: 
�
*train_cnn/ConvNet/conv1/HistogramSummary_2HistogramSummary.train_cnn/ConvNet/conv1/HistogramSummary_2/tagtrain_cnn/ConvNet/conv1/Relu*
T0*
_output_shapes
: 
|
.train_cnn/ConvNet/conv1/HistogramSummary_3/tagConst*
dtype0*
valueB Bconv1_maxpool*
_output_shapes
: 
�
*train_cnn/ConvNet/conv1/HistogramSummary_3HistogramSummary.train_cnn/ConvNet/conv1/HistogramSummary_3/tagtrain_cnn/ConvNet/conv1/MaxPool*
T0*
_output_shapes
: 
�
ConvNet/conv2/WVariable*
dtype0*
shape:@@*
	container *
shared_name *&
_output_shapes
:@@
�
/ConvNet/conv2/W/Initializer/random_normal/shapeConst*
dtype0*"
_class
loc:@ConvNet/conv2/W*%
valueB"      @   @   *
_output_shapes
:
�
.ConvNet/conv2/W/Initializer/random_normal/meanConst*
dtype0*"
_class
loc:@ConvNet/conv2/W*
valueB
 *    *
_output_shapes
: 
�
0ConvNet/conv2/W/Initializer/random_normal/stddevConst*
dtype0*"
_class
loc:@ConvNet/conv2/W*
valueB
 *o�:*
_output_shapes
: 
�
>ConvNet/conv2/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/ConvNet/conv2/W/Initializer/random_normal/shape*&
_output_shapes
:@@*
dtype0*
seed2!*

seed**
T0*"
_class
loc:@ConvNet/conv2/W
�
-ConvNet/conv2/W/Initializer/random_normal/mulMul>ConvNet/conv2/W/Initializer/random_normal/RandomStandardNormal0ConvNet/conv2/W/Initializer/random_normal/stddev*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
�
)ConvNet/conv2/W/Initializer/random_normalAdd-ConvNet/conv2/W/Initializer/random_normal/mul.ConvNet/conv2/W/Initializer/random_normal/mean*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
�
ConvNet/conv2/W/AssignAssignConvNet/conv2/W)ConvNet/conv2/W/Initializer/random_normal*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
�
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
�
!ConvNet/conv2/b/Initializer/ConstConst*
dtype0*"
_class
loc:@ConvNet/conv2/b*
valueB@*    *
_output_shapes
:@
�
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
�
train_cnn/ConvNet/conv2/Conv2DConv2Dtrain_cnn/ConvNet/conv1/MaxPoolConvNet/conv2/W/read*/
_output_shapes
:���������@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
train_cnn/ConvNet/conv2/addAddtrain_cnn/ConvNet/conv2/Conv2DConvNet/conv2/b/read*
T0*/
_output_shapes
:���������@
{
train_cnn/ConvNet/conv2/ReluRelutrain_cnn/ConvNet/conv2/add*
T0*/
_output_shapes
:���������@
�
train_cnn/ConvNet/conv2/MaxPoolMaxPooltrain_cnn/ConvNet/conv2/Relu*/
_output_shapes
:���������@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
z
,train_cnn/ConvNet/conv2/HistogramSummary/tagConst*
dtype0*
valueB Bconv2_weights*
_output_shapes
: 
�
(train_cnn/ConvNet/conv2/HistogramSummaryHistogramSummary,train_cnn/ConvNet/conv2/HistogramSummary/tagConvNet/conv2/W/read*
T0*
_output_shapes
: 
v
.train_cnn/ConvNet/conv2/HistogramSummary_1/tagConst*
dtype0*
valueB Bconv2_b*
_output_shapes
: 
�
*train_cnn/ConvNet/conv2/HistogramSummary_1HistogramSummary.train_cnn/ConvNet/conv2/HistogramSummary_1/tagConvNet/conv2/b/read*
T0*
_output_shapes
: 
x
.train_cnn/ConvNet/conv2/HistogramSummary_2/tagConst*
dtype0*
valueB B	conv2_out*
_output_shapes
: 
�
*train_cnn/ConvNet/conv2/HistogramSummary_2HistogramSummary.train_cnn/ConvNet/conv2/HistogramSummary_2/tagtrain_cnn/ConvNet/conv2/Relu*
T0*
_output_shapes
: 
|
.train_cnn/ConvNet/conv2/HistogramSummary_3/tagConst*
dtype0*
valueB Bconv2_maxpool*
_output_shapes
: 
�
*train_cnn/ConvNet/conv2/HistogramSummary_3HistogramSummary.train_cnn/ConvNet/conv2/HistogramSummary_3/tagtrain_cnn/ConvNet/conv2/MaxPool*
T0*
_output_shapes
: 
p
train_cnn/ConvNet/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
train_cnn/ConvNet/ReshapeReshapetrain_cnn/ConvNet/conv2/MaxPooltrain_cnn/ConvNet/Reshape/shape*(
_output_shapes
:���������� *
T0*
Tshape0
�
ConvNet/fc1/WVariable*
dtype0*
shape:
� �*
	container *
shared_name * 
_output_shapes
:
� �
�
-ConvNet/fc1/W/Initializer/random_normal/shapeConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB"   �  *
_output_shapes
:
�
,ConvNet/fc1/W/Initializer/random_normal/meanConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB
 *    *
_output_shapes
: 
�
.ConvNet/fc1/W/Initializer/random_normal/stddevConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB
 *o�:*
_output_shapes
: 
�
<ConvNet/fc1/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-ConvNet/fc1/W/Initializer/random_normal/shape* 
_output_shapes
:
� �*
dtype0*
seed2<*

seed**
T0* 
_class
loc:@ConvNet/fc1/W
�
+ConvNet/fc1/W/Initializer/random_normal/mulMul<ConvNet/fc1/W/Initializer/random_normal/RandomStandardNormal.ConvNet/fc1/W/Initializer/random_normal/stddev* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
� �
�
'ConvNet/fc1/W/Initializer/random_normalAdd+ConvNet/fc1/W/Initializer/random_normal/mul,ConvNet/fc1/W/Initializer/random_normal/mean* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
� �
�
ConvNet/fc1/W/AssignAssignConvNet/fc1/W'ConvNet/fc1/W/Initializer/random_normal*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
� �
z
ConvNet/fc1/W/readIdentityConvNet/fc1/W* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
� �
�
.ConvNet/fc1/W/Regularizer/l2_regularizer/scaleConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB
 *o�:*
_output_shapes
: 
�
/ConvNet/fc1/W/Regularizer/l2_regularizer/L2LossL2LossConvNet/fc1/W/read* 
_class
loc:@ConvNet/fc1/W*
T0*
_output_shapes
: 
�
(ConvNet/fc1/W/Regularizer/l2_regularizerMul.ConvNet/fc1/W/Regularizer/l2_regularizer/scale/ConvNet/fc1/W/Regularizer/l2_regularizer/L2Loss* 
_class
loc:@ConvNet/fc1/W*
T0*
_output_shapes
: 
y
ConvNet/fc1/bVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
ConvNet/fc1/b/Initializer/ConstConst*
dtype0* 
_class
loc:@ConvNet/fc1/b*
valueB�*    *
_output_shapes	
:�
�
ConvNet/fc1/b/AssignAssignConvNet/fc1/bConvNet/fc1/b/Initializer/Const*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:�
u
ConvNet/fc1/b/readIdentityConvNet/fc1/b* 
_class
loc:@ConvNet/fc1/b*
T0*
_output_shapes	
:�
�
train_cnn/ConvNet/fc1/MatMulMatMultrain_cnn/ConvNet/ReshapeConvNet/fc1/W/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
�
train_cnn/ConvNet/fc1/addAddtrain_cnn/ConvNet/fc1/MatMulConvNet/fc1/b/read*
T0*(
_output_shapes
:����������
p
train_cnn/ConvNet/fc1/ReluRelutrain_cnn/ConvNet/fc1/add*
T0*(
_output_shapes
:����������
v
*train_cnn/ConvNet/fc1/HistogramSummary/tagConst*
dtype0*
valueB Bfc1_weights*
_output_shapes
: 
�
&train_cnn/ConvNet/fc1/HistogramSummaryHistogramSummary*train_cnn/ConvNet/fc1/HistogramSummary/tagConvNet/fc1/W/read*
T0*
_output_shapes
: 
r
,train_cnn/ConvNet/fc1/HistogramSummary_1/tagConst*
dtype0*
valueB Bfc1_b*
_output_shapes
: 
�
(train_cnn/ConvNet/fc1/HistogramSummary_1HistogramSummary,train_cnn/ConvNet/fc1/HistogramSummary_1/tagConvNet/fc1/b/read*
T0*
_output_shapes
: 
t
,train_cnn/ConvNet/fc1/HistogramSummary_2/tagConst*
dtype0*
valueB Bfc1_out*
_output_shapes
: 
�
(train_cnn/ConvNet/fc1/HistogramSummary_2HistogramSummary,train_cnn/ConvNet/fc1/HistogramSummary_2/tagtrain_cnn/ConvNet/fc1/Relu*
T0*
_output_shapes
: 
�
ConvNet/fc2/WVariable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
-ConvNet/fc2/W/Initializer/random_normal/shapeConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB"�  �   *
_output_shapes
:
�
,ConvNet/fc2/W/Initializer/random_normal/meanConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB
 *    *
_output_shapes
: 
�
.ConvNet/fc2/W/Initializer/random_normal/stddevConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB
 *o�:*
_output_shapes
: 
�
<ConvNet/fc2/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-ConvNet/fc2/W/Initializer/random_normal/shape* 
_output_shapes
:
��*
dtype0*
seed2U*

seed**
T0* 
_class
loc:@ConvNet/fc2/W
�
+ConvNet/fc2/W/Initializer/random_normal/mulMul<ConvNet/fc2/W/Initializer/random_normal/RandomStandardNormal.ConvNet/fc2/W/Initializer/random_normal/stddev* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
��
�
'ConvNet/fc2/W/Initializer/random_normalAdd+ConvNet/fc2/W/Initializer/random_normal/mul,ConvNet/fc2/W/Initializer/random_normal/mean* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
��
�
ConvNet/fc2/W/AssignAssignConvNet/fc2/W'ConvNet/fc2/W/Initializer/random_normal*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
��
z
ConvNet/fc2/W/readIdentityConvNet/fc2/W* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
��
�
.ConvNet/fc2/W/Regularizer/l2_regularizer/scaleConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB
 *o�:*
_output_shapes
: 
�
/ConvNet/fc2/W/Regularizer/l2_regularizer/L2LossL2LossConvNet/fc2/W/read* 
_class
loc:@ConvNet/fc2/W*
T0*
_output_shapes
: 
�
(ConvNet/fc2/W/Regularizer/l2_regularizerMul.ConvNet/fc2/W/Regularizer/l2_regularizer/scale/ConvNet/fc2/W/Regularizer/l2_regularizer/L2Loss* 
_class
loc:@ConvNet/fc2/W*
T0*
_output_shapes
: 
y
ConvNet/fc2/bVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
ConvNet/fc2/b/Initializer/ConstConst*
dtype0* 
_class
loc:@ConvNet/fc2/b*
valueB�*    *
_output_shapes	
:�
�
ConvNet/fc2/b/AssignAssignConvNet/fc2/bConvNet/fc2/b/Initializer/Const*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:�
u
ConvNet/fc2/b/readIdentityConvNet/fc2/b* 
_class
loc:@ConvNet/fc2/b*
T0*
_output_shapes	
:�
�
train_cnn/ConvNet/fc2/MatMulMatMultrain_cnn/ConvNet/fc1/ReluConvNet/fc2/W/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
�
train_cnn/ConvNet/fc2/addAddtrain_cnn/ConvNet/fc2/MatMulConvNet/fc2/b/read*
T0*(
_output_shapes
:����������
p
train_cnn/ConvNet/fc2/ReluRelutrain_cnn/ConvNet/fc2/add*
T0*(
_output_shapes
:����������
v
*train_cnn/ConvNet/fc2/HistogramSummary/tagConst*
dtype0*
valueB Bfc2_weights*
_output_shapes
: 
�
&train_cnn/ConvNet/fc2/HistogramSummaryHistogramSummary*train_cnn/ConvNet/fc2/HistogramSummary/tagConvNet/fc2/W/read*
T0*
_output_shapes
: 
r
,train_cnn/ConvNet/fc2/HistogramSummary_1/tagConst*
dtype0*
valueB Bfc2_b*
_output_shapes
: 
�
(train_cnn/ConvNet/fc2/HistogramSummary_1HistogramSummary,train_cnn/ConvNet/fc2/HistogramSummary_1/tagConvNet/fc2/b/read*
T0*
_output_shapes
: 
t
,train_cnn/ConvNet/fc2/HistogramSummary_2/tagConst*
dtype0*
valueB Bfc2_out*
_output_shapes
: 
�
(train_cnn/ConvNet/fc2/HistogramSummary_2HistogramSummary,train_cnn/ConvNet/fc2/HistogramSummary_2/tagtrain_cnn/ConvNet/fc2/Relu*
T0*
_output_shapes
: 
�
ConvNet/logits/WVariable*
dtype0*
shape:	�
*
	container *
shared_name *
_output_shapes
:	�

�
0ConvNet/logits/W/Initializer/random_normal/shapeConst*
dtype0*#
_class
loc:@ConvNet/logits/W*
valueB"�   
   *
_output_shapes
:
�
/ConvNet/logits/W/Initializer/random_normal/meanConst*
dtype0*#
_class
loc:@ConvNet/logits/W*
valueB
 *    *
_output_shapes
: 
�
1ConvNet/logits/W/Initializer/random_normal/stddevConst*
dtype0*#
_class
loc:@ConvNet/logits/W*
valueB
 *o�:*
_output_shapes
: 
�
?ConvNet/logits/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0ConvNet/logits/W/Initializer/random_normal/shape*
_output_shapes
:	�
*
dtype0*
seed2n*

seed**
T0*#
_class
loc:@ConvNet/logits/W
�
.ConvNet/logits/W/Initializer/random_normal/mulMul?ConvNet/logits/W/Initializer/random_normal/RandomStandardNormal1ConvNet/logits/W/Initializer/random_normal/stddev*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
:	�

�
*ConvNet/logits/W/Initializer/random_normalAdd.ConvNet/logits/W/Initializer/random_normal/mul/ConvNet/logits/W/Initializer/random_normal/mean*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
:	�

�
ConvNet/logits/W/AssignAssignConvNet/logits/W*ConvNet/logits/W/Initializer/random_normal*
validate_shape(*#
_class
loc:@ConvNet/logits/W*
use_locking(*
T0*
_output_shapes
:	�

�
ConvNet/logits/W/readIdentityConvNet/logits/W*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
:	�

�
1ConvNet/logits/W/Regularizer/l2_regularizer/scaleConst*
dtype0*#
_class
loc:@ConvNet/logits/W*
valueB
 *o�:*
_output_shapes
: 
�
2ConvNet/logits/W/Regularizer/l2_regularizer/L2LossL2LossConvNet/logits/W/read*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
: 
�
+ConvNet/logits/W/Regularizer/l2_regularizerMul1ConvNet/logits/W/Regularizer/l2_regularizer/scale2ConvNet/logits/W/Regularizer/l2_regularizer/L2Loss*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
: 
z
ConvNet/logits/bVariable*
dtype0*
shape:
*
	container *
shared_name *
_output_shapes
:

�
"ConvNet/logits/b/Initializer/ConstConst*
dtype0*#
_class
loc:@ConvNet/logits/b*
valueB
*    *
_output_shapes
:

�
ConvNet/logits/b/AssignAssignConvNet/logits/b"ConvNet/logits/b/Initializer/Const*
validate_shape(*#
_class
loc:@ConvNet/logits/b*
use_locking(*
T0*
_output_shapes
:

}
ConvNet/logits/b/readIdentityConvNet/logits/b*#
_class
loc:@ConvNet/logits/b*
T0*
_output_shapes
:

�
train_cnn/ConvNet/logits/MatMulMatMultrain_cnn/ConvNet/fc2/ReluConvNet/logits/W/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������

�
train_cnn/ConvNet/logits/addAddtrain_cnn/ConvNet/logits/MatMulConvNet/logits/b/read*
T0*'
_output_shapes
:���������

|
-train_cnn/ConvNet/logits/HistogramSummary/tagConst*
dtype0*
valueB Blogits_weights*
_output_shapes
: 
�
)train_cnn/ConvNet/logits/HistogramSummaryHistogramSummary-train_cnn/ConvNet/logits/HistogramSummary/tagConvNet/logits/W/read*
T0*
_output_shapes
: 
x
/train_cnn/ConvNet/logits/HistogramSummary_1/tagConst*
dtype0*
valueB Blogits_b*
_output_shapes
: 
�
+train_cnn/ConvNet/logits/HistogramSummary_1HistogramSummary/train_cnn/ConvNet/logits/HistogramSummary_1/tagConvNet/logits/b/read*
T0*
_output_shapes
: 
z
/train_cnn/ConvNet/logits/HistogramSummary_2/tagConst*
dtype0*
valueB B
logits_out*
_output_shapes
: 
�
+train_cnn/ConvNet/logits/HistogramSummary_2HistogramSummary/train_cnn/ConvNet/logits/HistogramSummary_2/tagtrain_cnn/ConvNet/logits/add*
T0*
_output_shapes
: 
c
!train_cnn/cross-entropy-loss/RankConst*
dtype0*
value	B :*
_output_shapes
: 
~
"train_cnn/cross-entropy-loss/ShapeShapetrain_cnn/ConvNet/logits/add*
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
�
$train_cnn/cross-entropy-loss/Shape_1Shapetrain_cnn/ConvNet/logits/add*
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
�
 train_cnn/cross-entropy-loss/SubSub#train_cnn/cross-entropy-loss/Rank_1"train_cnn/cross-entropy-loss/Sub/y*
T0*
_output_shapes
: 
�
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
�
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
���������*
_output_shapes
:
�
#train_cnn/cross-entropy-loss/concatConcat.train_cnn/cross-entropy-loss/concat/concat_dim,train_cnn/cross-entropy-loss/concat/values_0"train_cnn/cross-entropy-loss/Slice*
_output_shapes
:*
T0*
N
�
$train_cnn/cross-entropy-loss/ReshapeReshapetrain_cnn/ConvNet/logits/add#train_cnn/cross-entropy-loss/concat*0
_output_shapes
:������������������*
T0*
Tshape0
e
#train_cnn/cross-entropy-loss/Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
e
$train_cnn/cross-entropy-loss/Shape_2Shapey*
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
�
"train_cnn/cross-entropy-loss/Sub_1Sub#train_cnn/cross-entropy-loss/Rank_2$train_cnn/cross-entropy-loss/Sub_1/y*
T0*
_output_shapes
: 
�
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
�
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
�
.train_cnn/cross-entropy-loss/concat_1/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
�
%train_cnn/cross-entropy-loss/concat_1Concat0train_cnn/cross-entropy-loss/concat_1/concat_dim.train_cnn/cross-entropy-loss/concat_1/values_0$train_cnn/cross-entropy-loss/Slice_1*
_output_shapes
:*
T0*
N
�
&train_cnn/cross-entropy-loss/Reshape_1Reshapey%train_cnn/cross-entropy-loss/concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
�
)train_cnn/cross-entropy-loss/crossentropySoftmaxCrossEntropyWithLogits$train_cnn/cross-entropy-loss/Reshape&train_cnn/cross-entropy-loss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
f
$train_cnn/cross-entropy-loss/Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
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
�
)train_cnn/cross-entropy-loss/Slice_2/sizePack"train_cnn/cross-entropy-loss/Sub_2*
N*
T0*
_output_shapes
:*

axis 
�
$train_cnn/cross-entropy-loss/Slice_2Slice"train_cnn/cross-entropy-loss/Shape*train_cnn/cross-entropy-loss/Slice_2/begin)train_cnn/cross-entropy-loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
&train_cnn/cross-entropy-loss/Reshape_2Reshape)train_cnn/cross-entropy-loss/crossentropy$train_cnn/cross-entropy-loss/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
l
"train_cnn/cross-entropy-loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
!train_cnn/cross-entropy-loss/lossMean&train_cnn/cross-entropy-loss/Reshape_2"train_cnn/cross-entropy-loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
/train_cnn/cross-entropy-loss/ScalarSummary/tagsConst*
dtype0*#
valueB Bcross-entropy loss*
_output_shapes
: 
�
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
�
train_cnn/accuracy/ArgMaxArgMaxy#train_cnn/accuracy/ArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
g
%train_cnn/accuracy/ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
train_cnn/accuracy/ArgMax_1ArgMaxtrain_cnn/ConvNet/logits/add%train_cnn/accuracy/ArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
�
train_cnn/accuracy/EqualEqualtrain_cnn/accuracy/ArgMaxtrain_cnn/accuracy/ArgMax_1*
T0	*#
_output_shapes
:���������
v
train_cnn/accuracy/CastCasttrain_cnn/accuracy/Equal*

DstT0*

SrcT0
*#
_output_shapes
:���������
b
train_cnn/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
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
�
 train_cnn/accuracy/ScalarSummaryScalarSummary%train_cnn/accuracy/ScalarSummary/tagstrain_cnn/accuracy/Mean*
T0*
_output_shapes
: 
�
#train_cnn/MergeSummary/MergeSummaryMergeSummary(train_cnn/ConvNet/conv1/HistogramSummary*train_cnn/ConvNet/conv1/HistogramSummary_1*train_cnn/ConvNet/conv1/HistogramSummary_2*train_cnn/ConvNet/conv1/HistogramSummary_3(train_cnn/ConvNet/conv2/HistogramSummary*train_cnn/ConvNet/conv2/HistogramSummary_1*train_cnn/ConvNet/conv2/HistogramSummary_2*train_cnn/ConvNet/conv2/HistogramSummary_3&train_cnn/ConvNet/fc1/HistogramSummary(train_cnn/ConvNet/fc1/HistogramSummary_1(train_cnn/ConvNet/fc1/HistogramSummary_2&train_cnn/ConvNet/fc2/HistogramSummary(train_cnn/ConvNet/fc2/HistogramSummary_1(train_cnn/ConvNet/fc2/HistogramSummary_2)train_cnn/ConvNet/logits/HistogramSummary+train_cnn/ConvNet/logits/HistogramSummary_1+train_cnn/ConvNet/logits/HistogramSummary_2*train_cnn/cross-entropy-loss/ScalarSummary train_cnn/accuracy/ScalarSummary*
_output_shapes
: *
N
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
 *  �?*
_output_shapes
: 
w
train_cnn/gradients/FillFilltrain_cnn/gradients/Shapetrain_cnn/gradients/Const*
T0*
_output_shapes
: 
�
Htrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ReshapeReshapetrain_cnn/gradients/FillHtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ShapeShape&train_cnn/cross-entropy-loss/Reshape_2*
out_type0*
T0*
_output_shapes
:
�
?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/TileTileBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Reshape@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_1Shape&train_cnn/cross-entropy-loss/Reshape_2*
out_type0*
T0*
_output_shapes
:
�
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
�
@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ProdProdBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_1@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Atrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Prod_1ProdBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_2Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
Dtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/MaximumMaximumAtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Prod_1Dtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
Ctrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/floordivDiv?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ProdBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Maximum*
T0*
_output_shapes
: 
�
?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/CastCastCtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/truedivDiv?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Tile?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Cast*
T0*#
_output_shapes
:���������
�
Etrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/ShapeShape)train_cnn/cross-entropy-loss/crossentropy*
out_type0*
T0*
_output_shapes
:
�
Gtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/ReshapeReshapeBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/truedivEtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
train_cnn/gradients/zeros_like	ZerosLike+train_cnn/cross-entropy-loss/crossentropy:1*
T0*0
_output_shapes
:������������������
�
Qtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
Mtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims
ExpandDimsGtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/ReshapeQtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Ftrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/mulMulMtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims+train_cnn/cross-entropy-loss/crossentropy:1*
T0*0
_output_shapes
:������������������
�
Ctrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ShapeShapetrain_cnn/ConvNet/logits/add*
out_type0*
T0*
_output_shapes
:
�
Etrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ReshapeReshapeFtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/mulCtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
;train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/ShapeShapetrain_cnn/ConvNet/logits/MatMul*
out_type0*
T0*
_output_shapes
:
�
=train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Shape_1Const*
dtype0*
valueB:
*
_output_shapes
:
�
Ktrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/BroadcastGradientArgsBroadcastGradientArgs;train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Shape=train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/SumSumEtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ReshapeKtrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
=train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/ReshapeReshape9train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Sum;train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
;train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Sum_1SumEtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ReshapeMtrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
?train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Reshape_1Reshape;train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Sum_1=train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0
�
Ftrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/group_depsNoOp>^train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Reshape@^train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Reshape_1
�
Ntrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/control_dependencyIdentity=train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/ReshapeG^train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/group_deps*P
_classF
DBloc:@train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Reshape*
T0*'
_output_shapes
:���������

�
Ptrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/control_dependency_1Identity?train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Reshape_1G^train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/group_deps*R
_classH
FDloc:@train_cnn/gradients/train_cnn/ConvNet/logits/add_grad/Reshape_1*
T0*
_output_shapes
:

�
?train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMulMatMulNtrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/control_dependencyConvNet/logits/W/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
Atrain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMul_1MatMultrain_cnn/ConvNet/fc2/ReluNtrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�

�
Itrain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/group_depsNoOp@^train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMulB^train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMul_1
�
Qtrain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/control_dependencyIdentity?train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMulJ^train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
Strain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/control_dependency_1IdentityAtrain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMul_1J^train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/group_deps*T
_classJ
HFloc:@train_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�

�
<train_cnn/gradients/train_cnn/ConvNet/fc2/Relu_grad/ReluGradReluGradQtrain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/control_dependencytrain_cnn/ConvNet/fc2/Relu*
T0*(
_output_shapes
:����������
�
8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/ShapeShapetrain_cnn/ConvNet/fc2/MatMul*
out_type0*
T0*
_output_shapes
:
�
:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
Htrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/BroadcastGradientArgsBroadcastGradientArgs8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/SumSum<train_cnn/gradients/train_cnn/ConvNet/fc2/Relu_grad/ReluGradHtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/ReshapeReshape6train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Sum8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Sum_1Sum<train_cnn/gradients/train_cnn/ConvNet/fc2/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
<train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1Reshape8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Sum_1:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
Ctrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/group_depsNoOp;^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape=^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1
�
Ktrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependencyIdentity:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/ReshapeD^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/group_deps*M
_classC
A?loc:@train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape*
T0*(
_output_shapes
:����������
�
Mtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependency_1Identity<train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1D^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
<train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMulMatMulKtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependencyConvNet/fc2/W/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
>train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1MatMultrain_cnn/ConvNet/fc1/ReluKtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
�
Ftrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul?^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1
�
Ntrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMulG^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
Ptrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1G^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
<train_cnn/gradients/train_cnn/ConvNet/fc1/Relu_grad/ReluGradReluGradNtrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependencytrain_cnn/ConvNet/fc1/Relu*
T0*(
_output_shapes
:����������
�
8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/ShapeShapetrain_cnn/ConvNet/fc1/MatMul*
out_type0*
T0*
_output_shapes
:
�
:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
Htrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/BroadcastGradientArgsBroadcastGradientArgs8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/SumSum<train_cnn/gradients/train_cnn/ConvNet/fc1/Relu_grad/ReluGradHtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/ReshapeReshape6train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Sum8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Sum_1Sum<train_cnn/gradients/train_cnn/ConvNet/fc1/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
<train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1Reshape8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Sum_1:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
Ctrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/group_depsNoOp;^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape=^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1
�
Ktrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependencyIdentity:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/ReshapeD^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/group_deps*M
_classC
A?loc:@train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape*
T0*(
_output_shapes
:����������
�
Mtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependency_1Identity<train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1D^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
<train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMulMatMulKtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependencyConvNet/fc1/W/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:���������� 
�
>train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1MatMultrain_cnn/ConvNet/ReshapeKtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
� �
�
Ftrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul?^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1
�
Ntrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMulG^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul*
T0*(
_output_shapes
:���������� 
�
Ptrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1G^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
� �
�
8train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/ShapeShapetrain_cnn/ConvNet/conv2/MaxPool*
out_type0*
T0*
_output_shapes
:
�
:train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/ReshapeReshapeNtrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependency8train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
Dtrain_cnn/gradients/train_cnn/ConvNet/conv2/MaxPool_grad/MaxPoolGradMaxPoolGradtrain_cnn/ConvNet/conv2/Relutrain_cnn/ConvNet/conv2/MaxPool:train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/Reshape*/
_output_shapes
:���������@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
>train_cnn/gradients/train_cnn/ConvNet/conv2/Relu_grad/ReluGradReluGradDtrain_cnn/gradients/train_cnn/ConvNet/conv2/MaxPool_grad/MaxPoolGradtrain_cnn/ConvNet/conv2/Relu*
T0*/
_output_shapes
:���������@
�
:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/ShapeShapetrain_cnn/ConvNet/conv2/Conv2D*
out_type0*
T0*
_output_shapes
:
�
<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape_1Const*
dtype0*
valueB:@*
_output_shapes
:
�
Jtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/BroadcastGradientArgsBroadcastGradientArgs:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/SumSum>train_cnn/gradients/train_cnn/ConvNet/conv2/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/ReshapeReshape8train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Sum:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Sum_1Sum>train_cnn/gradients/train_cnn/ConvNet/conv2/Relu_grad/ReluGradLtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1Reshape:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Sum_1<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
�
Etrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape?^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1
�
Mtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/ReshapeF^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape*
T0*/
_output_shapes
:���������@
�
Otrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1F^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1*
T0*
_output_shapes
:@
�
=train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/ShapeShapetrain_cnn/ConvNet/conv1/MaxPool*
out_type0*
T0*
_output_shapes
:
�
Ktrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/ShapeConvNet/conv2/W/readMtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
?train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"      @   @   *
_output_shapes
:
�
Ltrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltertrain_cnn/ConvNet/conv1/MaxPool?train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Shape_1Mtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency*&
_output_shapes
:@@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Htrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/group_depsNoOpL^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInputM^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilter
�
Ptrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependencyIdentityKtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInputI^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/group_deps*^
_classT
RPloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������@
�
Rtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependency_1IdentityLtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilterI^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/group_deps*_
_classU
SQloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
�
Dtrain_cnn/gradients/train_cnn/ConvNet/conv1/MaxPool_grad/MaxPoolGradMaxPoolGradtrain_cnn/ConvNet/conv1/Relutrain_cnn/ConvNet/conv1/MaxPoolPtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������  @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
>train_cnn/gradients/train_cnn/ConvNet/conv1/Relu_grad/ReluGradReluGradDtrain_cnn/gradients/train_cnn/ConvNet/conv1/MaxPool_grad/MaxPoolGradtrain_cnn/ConvNet/conv1/Relu*
T0*/
_output_shapes
:���������  @
�
:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/ShapeShapetrain_cnn/ConvNet/conv1/Conv2D*
out_type0*
T0*
_output_shapes
:
�
<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape_1Const*
dtype0*
valueB:@*
_output_shapes
:
�
Jtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/BroadcastGradientArgsBroadcastGradientArgs:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/SumSum>train_cnn/gradients/train_cnn/ConvNet/conv1/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/ReshapeReshape8train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Sum:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape*/
_output_shapes
:���������  @*
T0*
Tshape0
�
:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Sum_1Sum>train_cnn/gradients/train_cnn/ConvNet/conv1/Relu_grad/ReluGradLtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1Reshape:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Sum_1<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
�
Etrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape?^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1
�
Mtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/ReshapeF^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape*
T0*/
_output_shapes
:���������  @
�
Otrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1F^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1*
T0*
_output_shapes
:@
�
=train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/ShapeShapetrain_cnn/Reshape*
out_type0*
T0*
_output_shapes
:
�
Ktrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/ShapeConvNet/conv1/W/readMtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
?train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"         @   *
_output_shapes
:
�
Ltrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltertrain_cnn/Reshape?train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Shape_1Mtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency*&
_output_shapes
:@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Htrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/group_depsNoOpL^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInputM^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilter
�
Ptrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/control_dependencyIdentityKtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInputI^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/group_deps*^
_classT
RPloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������  
�
Rtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/control_dependency_1IdentityLtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilterI^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/group_deps*_
_classU
SQloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
�
#train_cnn/beta1_power/initial_valueConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *fff?*
_output_shapes
: 
�
train_cnn/beta1_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@ConvNet/conv1/W*
shared_name 
�
train_cnn/beta1_power/AssignAssigntrain_cnn/beta1_power#train_cnn/beta1_power/initial_value*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
�
train_cnn/beta1_power/readIdentitytrain_cnn/beta1_power*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
�
#train_cnn/beta2_power/initial_valueConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *w�?*
_output_shapes
: 
�
train_cnn/beta2_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@ConvNet/conv1/W*
shared_name 
�
train_cnn/beta2_power/AssignAssigntrain_cnn/beta2_power#train_cnn/beta2_power/initial_value*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
�
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
�
train_cnn/ConvNet/conv1/W/AdamVariable*
	container *&
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/W*
shared_name 
�
%train_cnn/ConvNet/conv1/W/Adam/AssignAssigntrain_cnn/ConvNet/conv1/W/Adamtrain_cnn/zeros*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
�
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
�
 train_cnn/ConvNet/conv1/W/Adam_1Variable*
	container *&
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/W*
shared_name 
�
'train_cnn/ConvNet/conv1/W/Adam_1/AssignAssign train_cnn/ConvNet/conv1/W/Adam_1train_cnn/zeros_1*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
�
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
�
train_cnn/ConvNet/conv1/b/AdamVariable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/b*
shared_name 
�
%train_cnn/ConvNet/conv1/b/Adam/AssignAssigntrain_cnn/ConvNet/conv1/b/Adamtrain_cnn/zeros_2*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
 train_cnn/ConvNet/conv1/b/Adam_1Variable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/b*
shared_name 
�
'train_cnn/ConvNet/conv1/b/Adam_1/AssignAssign train_cnn/ConvNet/conv1/b/Adam_1train_cnn/zeros_3*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
train_cnn/ConvNet/conv2/W/AdamVariable*
	container *&
_output_shapes
:@@*
dtype0*
shape:@@*"
_class
loc:@ConvNet/conv2/W*
shared_name 
�
%train_cnn/ConvNet/conv2/W/Adam/AssignAssigntrain_cnn/ConvNet/conv2/W/Adamtrain_cnn/zeros_4*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
�
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
�
 train_cnn/ConvNet/conv2/W/Adam_1Variable*
	container *&
_output_shapes
:@@*
dtype0*
shape:@@*"
_class
loc:@ConvNet/conv2/W*
shared_name 
�
'train_cnn/ConvNet/conv2/W/Adam_1/AssignAssign train_cnn/ConvNet/conv2/W/Adam_1train_cnn/zeros_5*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
�
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
�
train_cnn/ConvNet/conv2/b/AdamVariable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv2/b*
shared_name 
�
%train_cnn/ConvNet/conv2/b/Adam/AssignAssigntrain_cnn/ConvNet/conv2/b/Adamtrain_cnn/zeros_6*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
 train_cnn/ConvNet/conv2/b/Adam_1Variable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv2/b*
shared_name 
�
'train_cnn/ConvNet/conv2/b/Adam_1/AssignAssign train_cnn/ConvNet/conv2/b/Adam_1train_cnn/zeros_7*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
�
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
� �*    * 
_output_shapes
:
� �
�
train_cnn/ConvNet/fc1/W/AdamVariable*
	container * 
_output_shapes
:
� �*
dtype0*
shape:
� �* 
_class
loc:@ConvNet/fc1/W*
shared_name 
�
#train_cnn/ConvNet/fc1/W/Adam/AssignAssigntrain_cnn/ConvNet/fc1/W/Adamtrain_cnn/zeros_8*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
� �
�
!train_cnn/ConvNet/fc1/W/Adam/readIdentitytrain_cnn/ConvNet/fc1/W/Adam* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
� �
j
train_cnn/zeros_9Const*
dtype0*
valueB
� �*    * 
_output_shapes
:
� �
�
train_cnn/ConvNet/fc1/W/Adam_1Variable*
	container * 
_output_shapes
:
� �*
dtype0*
shape:
� �* 
_class
loc:@ConvNet/fc1/W*
shared_name 
�
%train_cnn/ConvNet/fc1/W/Adam_1/AssignAssigntrain_cnn/ConvNet/fc1/W/Adam_1train_cnn/zeros_9*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
� �
�
#train_cnn/ConvNet/fc1/W/Adam_1/readIdentitytrain_cnn/ConvNet/fc1/W/Adam_1* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
� �
a
train_cnn/zeros_10Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
train_cnn/ConvNet/fc1/b/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�* 
_class
loc:@ConvNet/fc1/b*
shared_name 
�
#train_cnn/ConvNet/fc1/b/Adam/AssignAssigntrain_cnn/ConvNet/fc1/b/Adamtrain_cnn/zeros_10*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:�
�
!train_cnn/ConvNet/fc1/b/Adam/readIdentitytrain_cnn/ConvNet/fc1/b/Adam* 
_class
loc:@ConvNet/fc1/b*
T0*
_output_shapes	
:�
a
train_cnn/zeros_11Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
train_cnn/ConvNet/fc1/b/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�* 
_class
loc:@ConvNet/fc1/b*
shared_name 
�
%train_cnn/ConvNet/fc1/b/Adam_1/AssignAssigntrain_cnn/ConvNet/fc1/b/Adam_1train_cnn/zeros_11*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:�
�
#train_cnn/ConvNet/fc1/b/Adam_1/readIdentitytrain_cnn/ConvNet/fc1/b/Adam_1* 
_class
loc:@ConvNet/fc1/b*
T0*
_output_shapes	
:�
k
train_cnn/zeros_12Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
train_cnn/ConvNet/fc2/W/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��* 
_class
loc:@ConvNet/fc2/W*
shared_name 
�
#train_cnn/ConvNet/fc2/W/Adam/AssignAssigntrain_cnn/ConvNet/fc2/W/Adamtrain_cnn/zeros_12*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
��
�
!train_cnn/ConvNet/fc2/W/Adam/readIdentitytrain_cnn/ConvNet/fc2/W/Adam* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
��
k
train_cnn/zeros_13Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
train_cnn/ConvNet/fc2/W/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��* 
_class
loc:@ConvNet/fc2/W*
shared_name 
�
%train_cnn/ConvNet/fc2/W/Adam_1/AssignAssigntrain_cnn/ConvNet/fc2/W/Adam_1train_cnn/zeros_13*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
��
�
#train_cnn/ConvNet/fc2/W/Adam_1/readIdentitytrain_cnn/ConvNet/fc2/W/Adam_1* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
��
a
train_cnn/zeros_14Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
train_cnn/ConvNet/fc2/b/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�* 
_class
loc:@ConvNet/fc2/b*
shared_name 
�
#train_cnn/ConvNet/fc2/b/Adam/AssignAssigntrain_cnn/ConvNet/fc2/b/Adamtrain_cnn/zeros_14*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:�
�
!train_cnn/ConvNet/fc2/b/Adam/readIdentitytrain_cnn/ConvNet/fc2/b/Adam* 
_class
loc:@ConvNet/fc2/b*
T0*
_output_shapes	
:�
a
train_cnn/zeros_15Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
train_cnn/ConvNet/fc2/b/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�* 
_class
loc:@ConvNet/fc2/b*
shared_name 
�
%train_cnn/ConvNet/fc2/b/Adam_1/AssignAssigntrain_cnn/ConvNet/fc2/b/Adam_1train_cnn/zeros_15*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:�
�
#train_cnn/ConvNet/fc2/b/Adam_1/readIdentitytrain_cnn/ConvNet/fc2/b/Adam_1* 
_class
loc:@ConvNet/fc2/b*
T0*
_output_shapes	
:�
i
train_cnn/zeros_16Const*
dtype0*
valueB	�
*    *
_output_shapes
:	�

�
train_cnn/ConvNet/logits/W/AdamVariable*
	container *
_output_shapes
:	�
*
dtype0*
shape:	�
*#
_class
loc:@ConvNet/logits/W*
shared_name 
�
&train_cnn/ConvNet/logits/W/Adam/AssignAssigntrain_cnn/ConvNet/logits/W/Adamtrain_cnn/zeros_16*
validate_shape(*#
_class
loc:@ConvNet/logits/W*
use_locking(*
T0*
_output_shapes
:	�

�
$train_cnn/ConvNet/logits/W/Adam/readIdentitytrain_cnn/ConvNet/logits/W/Adam*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
:	�

i
train_cnn/zeros_17Const*
dtype0*
valueB	�
*    *
_output_shapes
:	�

�
!train_cnn/ConvNet/logits/W/Adam_1Variable*
	container *
_output_shapes
:	�
*
dtype0*
shape:	�
*#
_class
loc:@ConvNet/logits/W*
shared_name 
�
(train_cnn/ConvNet/logits/W/Adam_1/AssignAssign!train_cnn/ConvNet/logits/W/Adam_1train_cnn/zeros_17*
validate_shape(*#
_class
loc:@ConvNet/logits/W*
use_locking(*
T0*
_output_shapes
:	�

�
&train_cnn/ConvNet/logits/W/Adam_1/readIdentity!train_cnn/ConvNet/logits/W/Adam_1*#
_class
loc:@ConvNet/logits/W*
T0*
_output_shapes
:	�

_
train_cnn/zeros_18Const*
dtype0*
valueB
*    *
_output_shapes
:

�
train_cnn/ConvNet/logits/b/AdamVariable*
	container *
_output_shapes
:
*
dtype0*
shape:
*#
_class
loc:@ConvNet/logits/b*
shared_name 
�
&train_cnn/ConvNet/logits/b/Adam/AssignAssigntrain_cnn/ConvNet/logits/b/Adamtrain_cnn/zeros_18*
validate_shape(*#
_class
loc:@ConvNet/logits/b*
use_locking(*
T0*
_output_shapes
:

�
$train_cnn/ConvNet/logits/b/Adam/readIdentitytrain_cnn/ConvNet/logits/b/Adam*#
_class
loc:@ConvNet/logits/b*
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

�
!train_cnn/ConvNet/logits/b/Adam_1Variable*
	container *
_output_shapes
:
*
dtype0*
shape:
*#
_class
loc:@ConvNet/logits/b*
shared_name 
�
(train_cnn/ConvNet/logits/b/Adam_1/AssignAssign!train_cnn/ConvNet/logits/b/Adam_1train_cnn/zeros_19*
validate_shape(*#
_class
loc:@ConvNet/logits/b*
use_locking(*
T0*
_output_shapes
:

�
&train_cnn/ConvNet/logits/b/Adam_1/readIdentity!train_cnn/ConvNet/logits/b/Adam_1*#
_class
loc:@ConvNet/logits/b*
T0*
_output_shapes
:

a
train_cnn/Adam/learning_rateConst*
dtype0*
valueB
 *��8*
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
 *w�?*
_output_shapes
: 
[
train_cnn/Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
/train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam	ApplyAdamConvNet/conv1/Wtrain_cnn/ConvNet/conv1/W/Adam train_cnn/ConvNet/conv1/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonRtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv1/W*
use_locking( *
T0*&
_output_shapes
:@
�
/train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam	ApplyAdamConvNet/conv1/btrain_cnn/ConvNet/conv1/b/Adam train_cnn/ConvNet/conv1/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonOtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv1/b*
use_locking( *
T0*
_output_shapes
:@
�
/train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam	ApplyAdamConvNet/conv2/Wtrain_cnn/ConvNet/conv2/W/Adam train_cnn/ConvNet/conv2/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonRtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv2/W*
use_locking( *
T0*&
_output_shapes
:@@
�
/train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam	ApplyAdamConvNet/conv2/btrain_cnn/ConvNet/conv2/b/Adam train_cnn/ConvNet/conv2/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonOtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv2/b*
use_locking( *
T0*
_output_shapes
:@
�
-train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam	ApplyAdamConvNet/fc1/Wtrain_cnn/ConvNet/fc1/W/Adamtrain_cnn/ConvNet/fc1/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonPtrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc1/W*
use_locking( *
T0* 
_output_shapes
:
� �
�
-train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam	ApplyAdamConvNet/fc1/btrain_cnn/ConvNet/fc1/b/Adamtrain_cnn/ConvNet/fc1/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonMtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc1/b*
use_locking( *
T0*
_output_shapes	
:�
�
-train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam	ApplyAdamConvNet/fc2/Wtrain_cnn/ConvNet/fc2/W/Adamtrain_cnn/ConvNet/fc2/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonPtrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc2/W*
use_locking( *
T0* 
_output_shapes
:
��
�
-train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam	ApplyAdamConvNet/fc2/btrain_cnn/ConvNet/fc2/b/Adamtrain_cnn/ConvNet/fc2/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonMtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc2/b*
use_locking( *
T0*
_output_shapes	
:�
�
0train_cnn/Adam/update_ConvNet/logits/W/ApplyAdam	ApplyAdamConvNet/logits/Wtrain_cnn/ConvNet/logits/W/Adam!train_cnn/ConvNet/logits/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonStrain_cnn/gradients/train_cnn/ConvNet/logits/MatMul_grad/tuple/control_dependency_1*#
_class
loc:@ConvNet/logits/W*
use_locking( *
T0*
_output_shapes
:	�

�
0train_cnn/Adam/update_ConvNet/logits/b/ApplyAdam	ApplyAdamConvNet/logits/btrain_cnn/ConvNet/logits/b/Adam!train_cnn/ConvNet/logits/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonPtrain_cnn/gradients/train_cnn/ConvNet/logits/add_grad/tuple/control_dependency_1*#
_class
loc:@ConvNet/logits/b*
use_locking( *
T0*
_output_shapes
:

�
train_cnn/Adam/mulMultrain_cnn/beta1_power/readtrain_cnn/Adam/beta10^train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam1^train_cnn/Adam/update_ConvNet/logits/W/ApplyAdam1^train_cnn/Adam/update_ConvNet/logits/b/ApplyAdam*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
�
train_cnn/Adam/AssignAssigntrain_cnn/beta1_powertrain_cnn/Adam/mul*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking( *
T0*
_output_shapes
: 
�
train_cnn/Adam/mul_1Multrain_cnn/beta2_power/readtrain_cnn/Adam/beta20^train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam1^train_cnn/Adam/update_ConvNet/logits/W/ApplyAdam1^train_cnn/Adam/update_ConvNet/logits/b/ApplyAdam*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
�
train_cnn/Adam/Assign_1Assigntrain_cnn/beta2_powertrain_cnn/Adam/mul_1*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking( *
T0*
_output_shapes
: 
�
train_cnn/AdamNoOp0^train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam1^train_cnn/Adam/update_ConvNet/logits/W/ApplyAdam1^train_cnn/Adam/update_ConvNet/logits/b/ApplyAdam^train_cnn/Adam/Assign^train_cnn/Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/save/tensor_namesConst*
dtype0*�
value�B� BConvNet/conv1/WBConvNet/conv1/bBConvNet/conv2/WBConvNet/conv2/bBConvNet/fc1/WBConvNet/fc1/bBConvNet/fc2/WBConvNet/fc2/bBConvNet/logits/WBConvNet/logits/bBtrain_cnn/ConvNet/conv1/W/AdamB train_cnn/ConvNet/conv1/W/Adam_1Btrain_cnn/ConvNet/conv1/b/AdamB train_cnn/ConvNet/conv1/b/Adam_1Btrain_cnn/ConvNet/conv2/W/AdamB train_cnn/ConvNet/conv2/W/Adam_1Btrain_cnn/ConvNet/conv2/b/AdamB train_cnn/ConvNet/conv2/b/Adam_1Btrain_cnn/ConvNet/fc1/W/AdamBtrain_cnn/ConvNet/fc1/W/Adam_1Btrain_cnn/ConvNet/fc1/b/AdamBtrain_cnn/ConvNet/fc1/b/Adam_1Btrain_cnn/ConvNet/fc2/W/AdamBtrain_cnn/ConvNet/fc2/W/Adam_1Btrain_cnn/ConvNet/fc2/b/AdamBtrain_cnn/ConvNet/fc2/b/Adam_1Btrain_cnn/ConvNet/logits/W/AdamB!train_cnn/ConvNet/logits/W/Adam_1Btrain_cnn/ConvNet/logits/b/AdamB!train_cnn/ConvNet/logits/b/Adam_1Btrain_cnn/beta1_powerBtrain_cnn/beta2_power*
_output_shapes
: 
�
save/save/shapes_and_slicesConst*
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
: 
�
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesConvNet/conv1/WConvNet/conv1/bConvNet/conv2/WConvNet/conv2/bConvNet/fc1/WConvNet/fc1/bConvNet/fc2/WConvNet/fc2/bConvNet/logits/WConvNet/logits/btrain_cnn/ConvNet/conv1/W/Adam train_cnn/ConvNet/conv1/W/Adam_1train_cnn/ConvNet/conv1/b/Adam train_cnn/ConvNet/conv1/b/Adam_1train_cnn/ConvNet/conv2/W/Adam train_cnn/ConvNet/conv2/W/Adam_1train_cnn/ConvNet/conv2/b/Adam train_cnn/ConvNet/conv2/b/Adam_1train_cnn/ConvNet/fc1/W/Adamtrain_cnn/ConvNet/fc1/W/Adam_1train_cnn/ConvNet/fc1/b/Adamtrain_cnn/ConvNet/fc1/b/Adam_1train_cnn/ConvNet/fc2/W/Adamtrain_cnn/ConvNet/fc2/W/Adam_1train_cnn/ConvNet/fc2/b/Adamtrain_cnn/ConvNet/fc2/b/Adam_1train_cnn/ConvNet/logits/W/Adam!train_cnn/ConvNet/logits/W/Adam_1train_cnn/ConvNet/logits/b/Adam!train_cnn/ConvNet/logits/b/Adam_1train_cnn/beta1_powertrain_cnn/beta2_power*)
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
�
save/restore_sliceRestoreSlice
save/Constsave/restore_slice/tensor_name"save/restore_slice/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
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
�
save/restore_slice_1RestoreSlice
save/Const save/restore_slice_1/tensor_name$save/restore_slice_1/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
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
�
save/restore_slice_2RestoreSlice
save/Const save/restore_slice_2/tensor_name$save/restore_slice_2/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
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
�
save/restore_slice_3RestoreSlice
save/Const save/restore_slice_3/tensor_name$save/restore_slice_3/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
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
�
save/restore_slice_4RestoreSlice
save/Const save/restore_slice_4/tensor_name$save/restore_slice_4/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_4AssignConvNet/fc1/Wsave/restore_slice_4*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
� �
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
�
save/restore_slice_5RestoreSlice
save/Const save/restore_slice_5/tensor_name$save/restore_slice_5/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_5AssignConvNet/fc1/bsave/restore_slice_5*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/restore_slice_6RestoreSlice
save/Const save/restore_slice_6/tensor_name$save/restore_slice_6/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_6AssignConvNet/fc2/Wsave/restore_slice_6*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
��
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
�
save/restore_slice_7RestoreSlice
save/Const save/restore_slice_7/tensor_name$save/restore_slice_7/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_7AssignConvNet/fc2/bsave/restore_slice_7*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:�
q
 save/restore_slice_8/tensor_nameConst*
dtype0*!
valueB BConvNet/logits/W*
_output_shapes
: 
e
$save/restore_slice_8/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_8RestoreSlice
save/Const save/restore_slice_8/tensor_name$save/restore_slice_8/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_8AssignConvNet/logits/Wsave/restore_slice_8*
validate_shape(*#
_class
loc:@ConvNet/logits/W*
use_locking(*
T0*
_output_shapes
:	�

q
 save/restore_slice_9/tensor_nameConst*
dtype0*!
valueB BConvNet/logits/b*
_output_shapes
: 
e
$save/restore_slice_9/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_9RestoreSlice
save/Const save/restore_slice_9/tensor_name$save/restore_slice_9/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_9AssignConvNet/logits/bsave/restore_slice_9*
validate_shape(*#
_class
loc:@ConvNet/logits/b*
use_locking(*
T0*
_output_shapes
:

�
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
�
save/restore_slice_10RestoreSlice
save/Const!save/restore_slice_10/tensor_name%save/restore_slice_10/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_10Assigntrain_cnn/ConvNet/conv1/W/Adamsave/restore_slice_10*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
�
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
�
save/restore_slice_11RestoreSlice
save/Const!save/restore_slice_11/tensor_name%save/restore_slice_11/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_11Assign train_cnn/ConvNet/conv1/W/Adam_1save/restore_slice_11*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
�
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
�
save/restore_slice_12RestoreSlice
save/Const!save/restore_slice_12/tensor_name%save/restore_slice_12/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_12Assigntrain_cnn/ConvNet/conv1/b/Adamsave/restore_slice_12*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/restore_slice_13RestoreSlice
save/Const!save/restore_slice_13/tensor_name%save/restore_slice_13/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_13Assign train_cnn/ConvNet/conv1/b/Adam_1save/restore_slice_13*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/restore_slice_14RestoreSlice
save/Const!save/restore_slice_14/tensor_name%save/restore_slice_14/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_14Assigntrain_cnn/ConvNet/conv2/W/Adamsave/restore_slice_14*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
�
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
�
save/restore_slice_15RestoreSlice
save/Const!save/restore_slice_15/tensor_name%save/restore_slice_15/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_15Assign train_cnn/ConvNet/conv2/W/Adam_1save/restore_slice_15*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
�
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
�
save/restore_slice_16RestoreSlice
save/Const!save/restore_slice_16/tensor_name%save/restore_slice_16/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_16Assigntrain_cnn/ConvNet/conv2/b/Adamsave/restore_slice_16*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/restore_slice_17RestoreSlice
save/Const!save/restore_slice_17/tensor_name%save/restore_slice_17/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
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
�
save/restore_slice_18RestoreSlice
save/Const!save/restore_slice_18/tensor_name%save/restore_slice_18/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_18Assigntrain_cnn/ConvNet/fc1/W/Adamsave/restore_slice_18*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
� �
�
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
�
save/restore_slice_19RestoreSlice
save/Const!save/restore_slice_19/tensor_name%save/restore_slice_19/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_19Assigntrain_cnn/ConvNet/fc1/W/Adam_1save/restore_slice_19*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
� �
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
�
save/restore_slice_20RestoreSlice
save/Const!save/restore_slice_20/tensor_name%save/restore_slice_20/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_20Assigntrain_cnn/ConvNet/fc1/b/Adamsave/restore_slice_20*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/restore_slice_21RestoreSlice
save/Const!save/restore_slice_21/tensor_name%save/restore_slice_21/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_21Assigntrain_cnn/ConvNet/fc1/b/Adam_1save/restore_slice_21*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/restore_slice_22RestoreSlice
save/Const!save/restore_slice_22/tensor_name%save/restore_slice_22/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_22Assigntrain_cnn/ConvNet/fc2/W/Adamsave/restore_slice_22*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
��
�
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
�
save/restore_slice_23RestoreSlice
save/Const!save/restore_slice_23/tensor_name%save/restore_slice_23/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_23Assigntrain_cnn/ConvNet/fc2/W/Adam_1save/restore_slice_23*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
��
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
�
save/restore_slice_24RestoreSlice
save/Const!save/restore_slice_24/tensor_name%save/restore_slice_24/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_24Assigntrain_cnn/ConvNet/fc2/b/Adamsave/restore_slice_24*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/restore_slice_25RestoreSlice
save/Const!save/restore_slice_25/tensor_name%save/restore_slice_25/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_25Assigntrain_cnn/ConvNet/fc2/b/Adam_1save/restore_slice_25*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:�
�
!save/restore_slice_26/tensor_nameConst*
dtype0*0
value'B% Btrain_cnn/ConvNet/logits/W/Adam*
_output_shapes
: 
f
%save/restore_slice_26/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_26RestoreSlice
save/Const!save/restore_slice_26/tensor_name%save/restore_slice_26/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_26Assigntrain_cnn/ConvNet/logits/W/Adamsave/restore_slice_26*
validate_shape(*#
_class
loc:@ConvNet/logits/W*
use_locking(*
T0*
_output_shapes
:	�

�
!save/restore_slice_27/tensor_nameConst*
dtype0*2
value)B' B!train_cnn/ConvNet/logits/W/Adam_1*
_output_shapes
: 
f
%save/restore_slice_27/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_27RestoreSlice
save/Const!save/restore_slice_27/tensor_name%save/restore_slice_27/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_27Assign!train_cnn/ConvNet/logits/W/Adam_1save/restore_slice_27*
validate_shape(*#
_class
loc:@ConvNet/logits/W*
use_locking(*
T0*
_output_shapes
:	�

�
!save/restore_slice_28/tensor_nameConst*
dtype0*0
value'B% Btrain_cnn/ConvNet/logits/b/Adam*
_output_shapes
: 
f
%save/restore_slice_28/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_28RestoreSlice
save/Const!save/restore_slice_28/tensor_name%save/restore_slice_28/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_28Assigntrain_cnn/ConvNet/logits/b/Adamsave/restore_slice_28*
validate_shape(*#
_class
loc:@ConvNet/logits/b*
use_locking(*
T0*
_output_shapes
:

�
!save/restore_slice_29/tensor_nameConst*
dtype0*2
value)B' B!train_cnn/ConvNet/logits/b/Adam_1*
_output_shapes
: 
f
%save/restore_slice_29/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_29RestoreSlice
save/Const!save/restore_slice_29/tensor_name%save/restore_slice_29/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_29Assign!train_cnn/ConvNet/logits/b/Adam_1save/restore_slice_29*
validate_shape(*#
_class
loc:@ConvNet/logits/b*
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
�
save/restore_slice_30RestoreSlice
save/Const!save/restore_slice_30/tensor_name%save/restore_slice_30/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
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
�
save/restore_slice_31RestoreSlice
save/Const!save/restore_slice_31/tensor_name%save/restore_slice_31/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_31Assigntrain_cnn/beta2_powersave/restore_slice_31*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31
�
initNoOp^ConvNet/conv1/W/Assign^ConvNet/conv1/b/Assign^ConvNet/conv2/W/Assign^ConvNet/conv2/b/Assign^ConvNet/fc1/W/Assign^ConvNet/fc1/b/Assign^ConvNet/fc2/W/Assign^ConvNet/fc2/b/Assign^ConvNet/logits/W/Assign^ConvNet/logits/b/Assign^train_cnn/beta1_power/Assign^train_cnn/beta2_power/Assign&^train_cnn/ConvNet/conv1/W/Adam/Assign(^train_cnn/ConvNet/conv1/W/Adam_1/Assign&^train_cnn/ConvNet/conv1/b/Adam/Assign(^train_cnn/ConvNet/conv1/b/Adam_1/Assign&^train_cnn/ConvNet/conv2/W/Adam/Assign(^train_cnn/ConvNet/conv2/W/Adam_1/Assign&^train_cnn/ConvNet/conv2/b/Adam/Assign(^train_cnn/ConvNet/conv2/b/Adam_1/Assign$^train_cnn/ConvNet/fc1/W/Adam/Assign&^train_cnn/ConvNet/fc1/W/Adam_1/Assign$^train_cnn/ConvNet/fc1/b/Adam/Assign&^train_cnn/ConvNet/fc1/b/Adam_1/Assign$^train_cnn/ConvNet/fc2/W/Adam/Assign&^train_cnn/ConvNet/fc2/W/Adam_1/Assign$^train_cnn/ConvNet/fc2/b/Adam/Assign&^train_cnn/ConvNet/fc2/b/Adam_1/Assign'^train_cnn/ConvNet/logits/W/Adam/Assign)^train_cnn/ConvNet/logits/W/Adam_1/Assign'^train_cnn/ConvNet/logits/b/Adam/Assign)^train_cnn/ConvNet/logits/b/Adam_1/Assign"��ek|�      �P�	�X�Mu�A*�
�
conv1_weights*�	   �xk�   @zq?     ��@! ���9H��);�w���s?2�
�N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ�[�=�k���*��ڽ���n�����豪}0ڰ�39W$:���.��fc���豪}0ڰ>��n����>5�"�g��>G&�$�>�[�=�k�>��~���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�
               @       @      @      1@      5@     �E@      J@     @P@      R@     �U@     �Z@     �\@     �\@     �`@     @]@     �Y@     @[@     �X@     �Y@     �W@     @R@      O@     �P@     �T@     �Q@      K@     �J@     �H@      E@     �B@     �B@      =@      B@      8@      >@      7@      .@      5@      7@      ;@      .@      3@      *@      @      @       @      @      &@      @      @      @      @      @      �?       @       @      @       @      @      �?      �?      @       @      �?              �?              �?      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              @               @      @      @       @              @              @       @      @      @      @      @      @      @      @       @      @      ,@      @      ,@      $@      *@       @      0@      ,@      1@      .@      :@      3@      4@      7@      C@      B@     �A@     �A@      A@      L@      J@     �F@     �N@     �P@     �T@     �W@     �S@     �X@     �T@     @V@     @W@     @U@     �Y@     @Y@     @Z@     �W@     @Y@      W@     @U@     @P@      O@      E@      7@       @      (@      �?      @              �?        
�
conv1_b*�	   `���   ��(�>      P@!  `p?��>)����*a�=2�G&�$��5�"�g���0�6�/n���u`P+d��豪}0ڰ���������?�ګ�;9��R����|�~���MZ��K���u��gr��R%������39W$:���X$�z��
�}�����4[_>������m!#���
�%W����ӤP���K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�ہkVl�p�w`f���n�=�.^ol�:�AC)8g�cR�k�e������0c�6��>?�J>������M>w&���qa>�����0c>cR�k�e>�����~>[#=�؏�>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�������:�              �?      �?      �?              �?      �?      @              �?      @              �?              �?              �?       @      �?              �?      �?      �?      �?               @              �?      �?              �?      �?              �?              �?      �?              �?              �?              �?      �?              �?       @      �?      �?      �?              �?              @      �?      �?      @      @      @       @      �?               @        
�
	conv1_out*�   �{�@     @�A!r��ߤiA)15!��YbA2�        �-���q=ڿ�ɓ�i>=�.^ol>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           pR%A               @              �?      �?      �?      �?      �?      �?              �?      �?      �?              @       @      @      @       @      @      @      @      @      @      @      @      @      &@      @      @      &@      $@      &@       @      (@      3@      (@      *@      $@      5@      5@      <@      6@     �@@     �C@     �A@      A@      F@     �A@      L@      J@     �O@      T@     �O@      Y@     �Z@     �Y@      \@     �\@     �a@     �a@     @c@      g@     �i@     �j@     �k@     r@     �q@     0r@     `v@     pw@     pz@     �}@     �|@      �@     Ѓ@     ȅ@     H�@     Ȉ@     ��@     p�@     Đ@     ̓@      �@     `�@     ��@     l�@     8�@     ��@     x�@     �@     D�@     �@     ��@     ج@     �@     H�@     w�@     s�@     �@     ƹ@     x�@     ��@     ��@     ?�@    ���@    ��@     �@    �b�@    ��@    @1�@    @��@     ��@    ���@    ���@    @��@    �:�@    ���@    `G�@    @�@    �
�@    `>�@     ��@    �I�@     %�@    ��@     ��@    pr�@     ��@    ���@    ���@    0s�@    �CA    h�A    ��A    H�A    �*	A    ��A    07A    \�A    `/A    0�A    �A    �A    D4A    �A    �>A    l !A    �~"A    L$A    ��%A    v�'A    4F)A    V$+A    -A    ��.A    �J0A    � 1A    �1A    ��2A    C!3A    a�3A    L�3A    թ3A    m^3A    ��2A    �1A    ��0A    ��.A    �w+A    ��'A    ��$A    �7!A    P`A    ��A    ̈A    �
A    �yA    P}�@    `��@    @6�@    @�@     p�@     ��@      P@      @        
�
conv1_maxpool*�   �{�@     @oA! ���UA)��@�tQA2�
        �-���q=39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>豪}0ڰ>��n����>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�
            �QA              �?      �?      @       @               @       @      �?              �?              �?      �?      @               @       @      �?      �?      @       @      @      @       @      @      &@      1@      @      @      @      $@      @      1@      5@      &@      *@      7@      :@      4@      7@      A@     �@@      >@     �G@      J@      J@     �F@     �Q@     �I@     @Q@      V@     @T@     �X@     @Y@     @X@     @Z@     �`@     @b@     �e@     �e@      h@     �l@     `n@     0r@     @r@     �r@     Ps@     �w@     �z@      }@     �@     h�@     H�@     �@     ��@     `�@      �@     ��@     Ȑ@      �@     T�@     ؖ@     Ț@     ̛@     T�@     >�@      �@     
�@     �@     V�@     ��@     ��@     ��@     �@     ��@     �@     �@     e�@     Z�@     ��@    ��@     ��@    �B�@    ���@    ���@    �,�@    �@�@    ��@    @�@     �@    �!�@     ��@    @��@    �+�@    `6�@     O�@     h�@    `L�@     ��@    ��@    �@�@    �S�@    Z�@    p]�@    ���@    ��@    ��@    � A    H�A    ��A    X�A    8�A    �7
A    ��A    ��A    ��A    �A    L�A    ��A    X A    d�A    rA     �A    �jA    �NA    ��A    ��A    TvA    duA    <�A    TvA    4cA    �kA    �<A    صA     �A    ��A    �@    p��@    �*�@    ��@     T�@    �5�@     �@     x�@      b@      @        
�
conv2_weights*�	   @hs�    ��p?      �@! �����?)�}֏E�?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP���u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x�cR�k�e������0c�4�j�6Z�Fixі�W���x��U�H��'ϱS�H��'ϱS>��x��U>4�j�6Z>��u}��\>ڿ�ɓ�i>=�.^ol>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              �?      @      ,@     �D@     @T@      f@     �u@     ��@     ��@     �@     �@     �@     ��@     <�@     ��@     �@     ��@     �@     �@     ʡ@     ��@     h�@     x�@     \�@     ��@     ��@     p�@     ��@     ��@     �@      �@     ��@      �@     x�@     p�@     (�@     ��@     �}@      |@      z@     �x@     �w@      v@     `q@     �p@     �n@     �j@      g@     `f@     `a@      d@     �`@     �`@     @Z@     �Y@     �Y@     @S@     �Q@     @P@     �S@     �L@     �M@      I@      H@     �H@      D@     �A@      >@      6@      :@      3@      7@      9@      4@      ,@      2@      @      (@      "@      $@      "@       @      @      @      "@      @      @       @      @      @      "@      �?      @      �?               @              �?       @      @               @      �?       @      �?       @               @              �?              �?               @              �?              �?              �?               @              �?              �?      �?      �?              �?      �?      @      �?              �?      �?              @      �?      �?      @       @      �?       @      �?      @      @      @      @      @      @      @      @      @      @       @      "@      @      $@       @       @      *@       @      3@      7@      ;@      8@      9@      ?@      >@      B@      E@      A@     �D@      I@      N@     @P@     @R@      S@     �S@      \@      W@     �Y@     @^@     �b@     �a@     �d@      d@      h@     @l@     �k@     �p@     `o@     �s@     �t@     �w@     �w@     Py@      ~@     ��@     ��@     ��@     ��@     @�@     ��@     @�@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     x�@     h�@     ��@     ��@     �@     ��@     Ȣ@     :�@     n�@     v�@     �@     ��@     �@     �@     X�@     p�@     Pu@      i@     �U@     �B@      @      �?        
�
conv2_b*�	    g��    2�?      P@!   ��?)!��?�[>2�1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;��|�~�>���]���>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?�������:�              �?       @       @      @       @      �?       @      �?      @              @      �?       @              �?      �?              �?              �?              �?              �?              �?      �?      �?       @      �?       @      �?      �?      @      �?       @      �?       @               @      @       @      @              �?       @      �?              �?        
�
	conv2_out*�   ����?     @oA!��р�:A)D���l��@2�
        �-���q=6��>?�J>������M>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�
            ��\A              �?              @      @      �?       @      �?      @              @      �?              �?      �?      �?      @      @       @      @      @       @      @      @      @      @       @      "@      $@      @      "@      $@      3@      .@      8@      .@      3@      4@      :@      8@      4@      ;@     �C@     �C@      E@      G@     �G@      K@     �L@     �T@     �S@     �T@     �T@     @X@      ]@     @Z@     �_@     �a@      e@      h@      g@     �i@     @m@     �n@     �s@      t@     @u@      u@     �y@     @}@     �@     ��@     P�@     ��@     �@     ��@     ��@     �@     P�@     ̒@     ܔ@     Ж@     ��@     �@     ��@     ^�@     �@     �@     n�@     4�@     T�@     Э@     ��@     �@     �@     ��@     ��@     2�@     E�@     ��@    �k�@     -�@     @�@    ���@    ���@     ��@     ��@    �%�@    @��@    @��@    ���@    �D�@     ��@     E�@    ���@    �F�@    �	�@    ��@    ���@    ���@     ��@    �A�@    p��@    ��@    `��@    P��@     ��@    �H�@     /�@    H� A    @{A    �4A    ��A    ��A    ��	A    x�A    ��A    �A    A    A    �A     �A    ��A    ,vA    ��A    A    ��A    lgA    d�A    ��A    �A     RA    ��A    �iA    X A    0��@     �@     ��@     h�@     F�@    �{�@    �t�@     ��@     ��@     P�@     @l@      B@      @        
�
conv2_maxpool*�   ����?     @OA!ۣ�*�@)���_�@2�	        �-���q=��x��U>Fixі�W>4�j�6Z>w&���qa>�����0c>:�AC)8g>ڿ�ɓ�i>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	            �/A              �?       @              @              �?               @              @       @               @      @      �?      @      �?       @      @      @      �?      @       @      @      &@      @      @      �?      @      "@      @      (@      $@      $@      .@      .@      ,@      0@      7@      2@      7@      .@      3@      =@      ?@     �A@      L@      A@      G@     �B@      H@      O@     �S@     �Q@     �Q@     �W@     �Z@      ^@     �`@     @`@     �b@     �c@     �`@     �f@     �c@      h@      p@     �p@     pu@     Pw@     �x@     `z@     �|@     p@     ��@     Ѓ@     Ѓ@     ȇ@     (�@     ��@     ��@      �@     ��@     ؕ@     ��@     �@     Ԛ@     ��@     �@     �@     .�@     <�@     �@     ��@     ��@     t�@     ̱@     ��@     ~�@     7�@     Һ@     o�@     ��@     �@     ��@     "�@    ��@     -�@    ���@    �.�@    ���@    ��@    ���@    @g�@    @��@    �O�@    @��@     %�@    �J�@    �H�@    ���@     g�@    @�@    �u�@    ��@    ���@    @��@    �Y�@    py�@    0d�@     K�@    @�@    �A A    (� A    �RA    P/A    � A    0s A    P�@    Pq�@     u�@    0��@    0��@    �$�@     ^�@    ���@    �]�@    �N�@     ��@     E�@     d�@     h�@     �r@      N@      @        
�%
fc1_weights*�$	    ��u�   `NIs?      8A!��R{�m�?)#fe�B'�?2�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6��so쩾4�7'_��+/��'v�V,����<�)��J��#���j�7'_��+/>_"s�$1>6NK��2>�`�}6D>��Ő�;F>��8"uH>6��>?�J>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?      "@     �C@      d@     `~@      �@     R�@     �@     ��@     �@    @��@    �9�@    �U�@    @�@    ��@    ���@     ��@     ��@    �n�@    @�@    @��@    ���@    ���@    @[�@    �R�@    @`�@     ��@     ��@    �M�@     ��@    @��@     ��@    �&�@     ��@    �q�@    ���@    ���@     !�@     3�@     @�@     ݹ@     ��@     s�@     ��@     ��@     İ@     ��@     X�@     �@     l�@     Σ@     ��@     v�@     ��@     ��@     ��@     p�@     Е@     P�@     ��@     ��@     �@     ��@      �@     ��@      �@     �@     �@     �~@     z@      z@     pv@     �s@      r@      q@     �m@     `o@     �i@     �f@     `h@     �d@     �a@     @Z@     @\@      X@     �U@     �Q@      T@     �T@     �R@     �O@     �P@     �D@      C@      I@     �@@      =@      ?@      A@      1@      5@      >@      1@      .@      &@      3@      .@      1@      (@      ,@      @      (@       @      @       @      @      @      @      @      @      @      @      @              @      @      �?      �?      �?      @      �?      �?              �?       @              �?              �?      �?               @      �?       @              �?      �?              �?               @      �?              �?              �?              �?              @       @       @      @      @      �?      �?      �?      @      @      @      @      @      �?      @      @      &@      @      @      $@      $@      "@      @      @      (@      (@      ,@      (@      *@      6@      4@      5@      0@      <@     �@@      F@      D@     �G@      A@     �D@     �F@     @Q@      P@     �Q@     �V@     @U@     �W@     @T@     �Z@     @^@      `@      b@     �f@      e@     �i@     �m@     �m@     �o@     0r@     @u@     �u@     �z@     0z@     �|@     ��@     Ѐ@     ��@     h�@     �@     p�@     �@     ��@     T�@     X�@     �@     ȗ@     @�@     ��@     ��@     T�@     ��@     Ҥ@     t�@     6�@     ~�@     ��@     D�@     ��@     ĳ@     z�@     ޷@     ��@     ��@     %�@     �@    �$�@     ��@    ���@     ��@     a�@     3�@    @a�@    ���@    @��@    @�@    ���@     ��@     1�@    ���@     5�@    ���@     ��@    �8�@    ��@    `��@    ���@    ���@    ���@    @��@    @T�@     �@    ���@     ��@    ���@     ��@     �@     �@     ��@     �c@      B@      $@        
�

fc1_b*�
	   ��T�   �R�?      x@!  p=:�?)�g*^��>2���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;;�"�qʾ�XQ�þ��~��¾        �-���q=�����0c>cR�k�e>��z!�?�>��ӤP��>��n����>�u`P+d�>5�"�g��>G&�$�>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�������:�              �?      *@      9@      &@      5@      3@      $@      *@      @      "@       @      @      @       @      @       @      �?              @       @      @       @              @      �?              �?              �?              �?              �?      �?              �?              :@              �?              �?              �?              �?              �?              @              �?              �?              �?               @               @      �?       @      @      @      @      @       @      @      @      $@      "@      $@      @      *@      &@      2@      1@      6@      0@       @        
�
fc1_out*�    �I�?     pA!(���l@)6�����?2�        �-���q=��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�            �A              �?              �?              �?      �?              �?       @              �?       @              �?              �?       @      @              �?      @              @       @       @       @      �?       @       @      @      @      �?      @      @      "@      �?      ,@      "@       @       @      4@      (@      0@      2@      1@      0@      2@      6@      :@      4@      D@     �B@      F@      O@      A@      N@     �M@     �P@      R@     �Q@     �T@     �X@     �X@     @\@     `a@     �a@     �c@      e@     �c@     @j@     �m@     �o@     �q@     s@     s@     `w@     @y@     �|@     �|@     �@     ��@     ؄@     ��@     Ј@     P�@     ��@     T�@     t�@     ��@     ��@     �@     8�@     ��@     P�@     N�@     �@     v�@     �@     "�@     �@     �@     �@     E�@     ;�@     ?�@     ��@     A�@     H�@     �@     ڻ@     R�@     a�@     ��@     ��@     ��@    ��@     ��@     N�@     *�@     r�@     :�@     A�@     ��@     ��@     ܢ@     �@     (�@     ��@     0|@     �h@     �V@      C@      @      �?        
�
fc2_weights*�	   �us�   �0�p?      �@!�wNu��)�{m���?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v������0c�w&���qa�6NK��2>�so쩾4>��Ő�;F>��8"uH>:�AC)8g>ڿ�ɓ�i>�i����v>E'�/��x>f^��`{>�����~>��ӤP��>�
�%W�>.��fc��>39W$:��>R%�����>�u��gr�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              �?      @      $@      5@     �O@     `b@     �n@     �y@      �@     ��@     ȏ@     �@     ԕ@     ��@     ��@     p�@     (�@     Ԛ@     l�@     ��@     ��@     ܖ@     ��@     �@     �@     ��@     X�@     P�@     ��@     P�@     `�@     ؄@     ��@     h�@     h�@      |@      w@     �x@     �v@     �r@     �r@     �p@     �l@     �k@     �j@      e@     �c@     �a@     �^@     �`@      _@     �W@     @X@      S@     �Q@     �P@     �Q@      O@     �N@      C@     �H@      F@      C@      9@      3@      =@      9@      8@      5@      1@      8@      2@      @      .@      "@       @      (@      @      @      @      @      @      @      @      @      @      @       @      @      �?      �?       @      @      @       @      @      �?      �?      �?      �?               @      �?      @      �?      �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?               @              �?      �?      �?              @       @       @      �?      @      �?      @      @      @      @      @      @      @      @      "@       @       @      (@      @      $@      $@      *@      0@      .@      1@      ?@      9@      =@      @@      6@      A@      ?@      G@      I@      K@      L@      M@      O@     �Q@     @S@     �V@      U@     �W@     �Z@     �Y@     �_@     ``@     �d@     �f@     �i@     �j@     @m@     `q@     �o@     �r@      w@      v@     �y@     @|@     �@     X�@     ��@     ��@     �@     `�@     Ȋ@     ��@     ��@     P�@     T�@     ��@     ��@     X�@     ��@      �@     \�@     �@     ��@     8�@     �@     ؗ@     ��@     \�@     x�@     ��@     `�@     �v@     �q@     �a@     @P@      4@      (@       @        
�
fc2_b*�	   ��.�   `h-?      h@!  @�j�Y?)�����>2��.����ڋ��vV�R9�8K�ߝ�a�Ϭ(���(��澢f����        �-���q=x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�������:�             �O@      0@              �?              �?              ,@               @               @      @      .@     @R@        
�

fc2_out*�
   ��4+?     pA!�\�FX$@)��n2�R?2�        �-���q=����W_>>p��Dp�@>Fixі�W>4�j�6Z>:�AC)8g>ڿ�ɓ�i>E'�/��x>f^��`{>�����~>�
�%W�>���m!#�>�4[_>��>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?�������:�            ,�@              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?       @      @       @      @      @      �?       @      @       @      @      @       @      @      @      �?      @      @      @      @      @      @       @      @      @      @      @      @      $@      @      @      &@       @      "@      (@      $@      &@      6@      =@     �G@     �K@      O@     @Y@     �U@     �c@     @j@     �n@     �t@     pv@     �|@     ��@     @�@     ��@     �@     ��@    �m�@     U�@     ��@    �5�@    �<�@     ��@     ��@     H�@     `s@      ,@        
�
logits_weights*�	   ���m�   ��j?      �@! ��W�?)�z���'^?2�	;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE�����_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ��n�����豪}0ڰ���������?�ګ���|�~���MZ��K����~���>�XQ��>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�������:�	              �?      �?       @      @      @      @      1@      0@      :@      >@      ;@      8@      >@     �B@      O@      @@     �A@     �F@      @@      F@      A@     �F@     �B@      ;@      =@      2@      <@      ,@      2@      1@      .@      &@      "@       @       @      *@       @      *@      @       @      @      @      @      @      @      @      @      @      @      @      @      @              @      �?       @               @       @      �?      @              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?               @              �?      �?               @              �?               @              �?      �?      �?               @      �?      �?      @      @      �?      @      @       @      �?      @      @       @      @      @      @      @      @      "@      (@      *@      @      &@      0@      3@      &@      2@      (@      ;@      0@      2@      4@      7@      =@     �G@      @@      C@      A@     �D@      D@      I@     �I@      E@     �A@      H@      B@     �J@     �C@      9@      <@      2@      "@      @      �?      @        
�
logits_b*�	   ��6�   ��6?      $@!      �=)�X�y��z>2(�.����ڋ���ڋ?�.�?�������:(              @              @        
�

logits_out*�	   ��B�   ��h?     ��@!   ��zw?)��9�B?2(�.����ڋ���ڋ?�.�?�������:(             ��@             ��@        

cross-entropy loss�]@

accuracy�"�=�j�r�      ����	��2Uu�A
*�
�
conv1_weights*�	   �� p�   ��p?     ��@!  Sɥ�)O|4�=�z?2�
;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�MZ��K���u��gr��R%������39W$:���.��fc�����n����>�u`P+d�>�*��ڽ>�[�=�k�>
�/eq
�>;�"�q�>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�
              @       @      *@      6@     �F@     �K@     �Q@     �W@      ^@      [@     �`@     ``@     �]@     �]@     �Z@     @[@     @]@     @]@      Z@     �W@      X@      T@      Q@     �O@     �S@      E@      E@     �F@      =@      B@      D@      C@      C@      7@      5@      9@      6@      ,@      ;@      &@      3@      @      *@      $@      "@      "@      @      $@      @      &@      @      @       @      @      @      @      @      @      �?               @      @       @      @              �?      �?      �?      �?      �?              �?      �?      �?              �?              �?              �?      �?              �?              �?               @               @              �?      @      �?      �?       @              @      @       @       @      @      @      �?       @      @       @      &@      @      @      @       @      @      @      @      @      *@      ,@      ,@      1@      1@      6@      5@      1@      3@      8@     �D@     �@@     �B@      A@     �E@      A@     �K@     �J@     �K@     @S@      O@     �S@     �Q@     �R@     �S@     �S@     @U@      Z@     �U@     �\@     @V@     �T@      T@     @T@      P@     �G@      F@      ;@      1@      &@      @       @      �?        
�
conv1_b*�	   `^��>   �ɏ%?      P@!  �!z?)�v�"�6�>2�O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�������:�              �?              �?       @      @              @      @      �?       @       @      @      (@      *@      @      @       @      �?        
�
	conv1_out*�   @�!@     @�A!tK�(��A)B~�YܞA2�        �-���q=p
T~�;>����W_>>������M>28���FP>��u}��\>d�V�_>T�L<�>��z!�?�>�4[_>��>
�}���>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�������:�           0��~A              �?              �?              �?               @              �?              �?              �?      �?               @      @      �?       @      �?      �?      @       @      �?       @       @      �?      @      @       @      @      @      @      @      @      @      "@      @      @      *@      "@      *@      $@      $@      7@      4@      1@      1@      &@      6@      =@      <@      5@     �D@     �F@      C@     �F@     �M@     �L@     �O@      M@      O@      V@     @R@     @Y@     �`@     �]@     �_@     �b@     �c@     @e@     �f@     �j@     �l@     `o@     p@     �t@     �u@     �w@     `w@     }@     ��@     ��@      �@     ��@     x�@      �@     `�@     x�@      �@     l�@     ��@     ԕ@     ��@     D�@     �@      �@     n�@     B�@     ��@     ��@     ĩ@     X�@     ��@     <�@      �@     ��@     ��@     �@     b�@     �@    ���@    �%�@     <�@    ���@    �L�@    ���@     i�@    @�@    @��@    @��@    ���@     ��@    ��@    ���@    @��@     e�@     ��@    ���@    �)�@    @`�@    `�@    ���@     ��@    �v�@    p^�@    l�@    P��@    0,�@    ���@    �l A    A    8�A    p�A    ��A    O
A    ��A    (�A    pA    �*A    \�A    �A    <%A    ��A    hA    q A    ��!A    ؖ#A    2E%A    �'A    ��(A    `+A    �#-A    �I/A    Ժ0A    �1A    �2A    (�3A    ٧4A    �V5A    #�5A    �
6A    �6A    ބ5A    �y4A    �3A    �^1A    �.A    n�(A    8^#A    @�A    A    ��	A    `��@    ���@    �f�@    �$�@     \�@        
�
conv1_maxpool*�   @�!@     @oA!h]57PqA)��ʜ�k�A2�        �-���q=T�L<�>��z!�?�>�4[_>��>
�}���>X$�z�>.��fc��>���]���>�5�L�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�������:�           @�WA               @               @               @              �?              �?               @              @              �?               @      @       @      @      �?      �?              @              @      @       @      @              @      �?      �?      @      @      "@      @      @      *@       @      &@      0@      1@      2@      *@      0@      9@      1@      2@      F@      A@      8@      :@     �D@      B@      I@      F@      H@     �M@     @P@     @U@      T@     �Y@     �Q@     �^@     �\@      ^@     @`@     `b@      e@     �j@      i@     `l@     �l@     �l@     �r@     @v@     �w@     Py@      |@     ��@     ��@     �@     ��@     �@     ��@     x�@     �@     ��@     T�@     �@     p�@     @�@     d�@     ؜@     ̟@     `�@     ��@     �@     4�@      �@     N�@     ڮ@     k�@     �@     �@     Ƕ@     H�@     �@     ��@     �@     �@     ��@     k�@     ��@     o�@    ���@    �q�@    ���@     ��@    ��@     ��@     7�@    �#�@     �@     ��@    �9�@    �S�@    �\�@    ��@     w�@     =�@    `Q�@    �@    ���@    ���@    �&�@    ��@    �L�@    � A     WA    ((A    0�A    �A    �I
A    ��A    h:A    �A    `aA    4�A    8�A    �A    <A    ��A    HA    �	A    L�A    �9A    ��A    ��A     #A    �sA    ��A    �(A    �	A     �A    `��@    @|�@    @��@     ��@     ��@      �@        
�
conv2_weights*�	   �dRs�   ���r?      �@!P���P�-@)U9��O}�?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�4�j�6Z>��u}��\>w&���qa>�����0c>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>f^��`{>�����~>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?      @      *@      A@     �Q@      b@     @s@     �|@     ��@     P�@     ȑ@     ��@      �@     <�@     �@     `�@     ��@     0�@     |�@     P�@     ؝@     $�@     4�@     H�@     ��@     �@     ȓ@     ��@     А@     (�@     ��@     ��@      �@     @�@     p�@     ��@     ��@     �@     �|@     x@     py@     �t@     �s@      r@     0q@     �l@     �h@     `i@     `f@     �^@     �_@      `@     �\@     �X@     �^@     @S@     �T@     �Q@      T@     �Q@      E@      G@     �E@      H@      H@      @@      6@      >@      8@      =@      5@      5@      0@      1@      .@      .@      "@       @      $@      (@      "@      @       @      @      @      @      @      @      @      @       @       @      �?      �?      �?       @               @       @      �?      @       @              �?              �?               @              �?              �?              �?              �?              �?      �?               @              @              �?               @       @              @      @       @      @       @      @       @      @      �?      @              @       @      @       @      "@      @      @      @      @      @      @      &@      @      ,@      *@      $@      1@      1@      3@      3@      6@     �@@      :@      :@      A@      A@     �D@     �G@     �K@     �M@     �N@     �L@     @S@      W@     �X@     @V@     �_@     @]@     �a@      d@     �c@     �h@     �g@      j@      m@     �p@     �q@     �s@     �u@     �y@     Pz@     �~@      �@     0�@     H�@     X�@     �@     ��@     x�@     H�@     ��@     <�@      �@     <�@     ��@     \�@     ��@     $�@     ��@     2�@     ܢ@     Z�@     h�@     z�@     4�@     Ԥ@     &�@     ��@     ʠ@     �@      �@      �@     �@      �@     �t@      f@      S@      7@      $@      �?        
�
conv2_b*�	   `�D#�    ӋA?      P@!  P$\�?)c���v��>2�U�4@@�$��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�x?�x��>h�'��6�]���1��a˲�I��P=��pz�w�7���uE����>�f����>6�]��?����?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�������:�              �?      �?       @      �?              �?               @              �?              �?              �?              �?               @               @              �?       @       @      �?      @              &@      @      @      �?      @      @      @       @      �?        
�
	conv2_out*�   `Z�
@     @oA!���PA)�H�(ĝEA2�        �-���q=w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>BvŐ�r>E'�/��x>f^��`{>K���7�>u��6
�>T�L<�>�
�%W�>���m!#�>39W$:��>R%�����>�u��gr�>�MZ��K�>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�������:�           ���SA              �?      �?              �?              �?      �?              �?               @      �?              �?              �?      �?      @               @      @      @      @      @      @      @      @      "@       @      �?      @      @      @      @      @      "@      (@      (@      "@      5@      &@      &@      1@      1@      ,@      7@      :@      ?@     �A@      @@      ;@     �D@     �K@     �K@     �J@     �Q@     �K@     @Q@     �P@      W@     �S@     �Z@      Y@      `@     �Z@     `b@     �c@      f@     �f@      m@     @n@     �p@     0s@     r@     �w@     py@     |@     �~@      �@      �@     ��@     0�@     ؇@     �@     ��@      �@     �@     P�@     �@     L�@     ��@     T�@     $�@     ��@     ��@     x�@     `�@     ��@     ��@     ��@     ��@     x�@     ��@     ~�@     ��@     8�@     ڻ@     ��@    �U�@     ��@     x�@    ���@    �I�@     W�@    ���@    �i�@    @��@    @��@     ��@    @u�@     ��@     h�@    �$�@    ���@    `��@    @O�@     �@    �>�@    @��@    ���@    ���@    `[�@    `��@    P��@    @��@    `��@    ���@    ��@    ���@    �$A    `�A     A    X�A    ��A    �y	A    h�A    8�A    H	A    <A    �cA    ��A    ��A    H*A    �eA    ��A    �A    �0A    ܩA    ��A    �}A    |�A    <
A    ,�A    l�A    ��A    ԇA    A    �+A    �"A    0'A    p��@    �Q�@     M�@    ���@    @J�@     ��@     �@     ��@     ��@     `k@      G@        
�
conv2_maxpool*�   `Z�
@     @OA!R8vC��5A)B��>1A2�
        �-���q=�
�%W�>���m!#�>�5�L�>;9��R�>���?�ګ>豪}0ڰ>��n����>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�������:�
            ��,A              �?               @       @              �?              �?      �?              �?       @      @      @       @      @      @       @      &@      �?       @      @      @      @      @       @      @       @      @       @      @      3@      *@       @      8@      0@      2@      4@      5@      .@      ?@      1@      :@      4@      G@     �J@     �E@      I@     �H@     �D@     �P@     @R@     �R@     �X@     @W@     �_@     �`@     @_@      a@     `d@     �e@     �k@     �h@     �k@      o@     �q@     pr@     ps@     `t@     {@     �z@     �|@     ��@     �@     �@     P�@     ��@     �@     ��@     `�@     ��@     �@     �@     �@     ��@     ̚@     �@     Ԡ@     P�@     J�@     �@     ڨ@     ~�@     ��@     Я@     �@     ��@     �@     �@     f�@     �@     ��@     5�@    ��@     ��@     ��@     ��@    �F�@    ��@     #�@     ��@    ��@    ���@    ���@    ���@    �$�@     ��@    �Y�@    @}�@    `��@    @��@     U�@    ���@    `_�@    �?�@    �E�@    `��@    �(�@     ��@     Z�@    ��@    �f�@    p,�@    P��@    ���@    �y�@    ��@    p��@    ���@    �  A    ��@     '�@    p��@    P�@    0��@    ��@    �#�@    ���@    `��@    ���@    ���@    ���@    �o�@     ��@     "�@     8�@     �@     ��@     @j@     �P@        
�&
fc1_weights*�&	   ��u�    t?      8A!���y�J@)/������?2�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�u 5�9��z��6��so쩾4���-�z�!�%����Łt�=	���R�����#���j�Z�TA[����
"
ֽ�|86	Խ��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�z��6>u 5�9>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?      @     �F@      a@     �~@     �@     �@     "�@    ��@    �&�@     ��@    �"�@    @z�@     ��@    @��@    `O�@    ���@    ��@    ���@    ���@    `H�@    ��@    ��@    �w�@    @��@     �@     O�@     ��@     ��@    @��@    ���@     m�@    ���@     ~�@    �r�@    ���@    �o�@     ��@     �@     �@     ��@     u�@     �@     ��@     R�@     �@     P�@     ª@     ��@     `�@     ��@     @�@     ��@     ��@     �@     l�@     Е@     �@     0�@     @�@     �@     P�@     X�@     ��@     X�@     ��@     �@     @�@     �|@     `{@     y@      w@     �t@     �q@     `l@     �o@     `l@     �j@     �h@     @d@     @`@     �b@     �^@     �Z@     �^@     �S@     @W@      P@     �O@     �M@      N@      J@      G@     �@@      A@      :@     �C@      5@      9@      ;@      ;@      3@      7@      .@      .@      *@      "@      $@      "@      @      @      *@      @      $@      @       @      @      @      @      �?      @      @       @      @       @       @      @      �?              �?              �?       @      �?      �?      �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @              �?               @      @      �?       @              @      �?       @              �?      @      @      @       @      @      @      @      @      @      (@      @      "@      @      $@      ,@       @      .@      ,@      0@      6@      3@      4@      7@      5@      7@      4@     �E@      D@      B@      B@      G@     �E@     �G@     �J@     �O@      R@     �U@     �U@     �W@     @\@     @Z@     `b@     �b@      c@     @f@     `f@     `i@     Pp@      r@     �r@     s@     �w@     �w@     py@     �z@     �~@     �@     @�@     @�@     ��@     ��@     ؋@     0�@     ��@     <�@     D�@     ��@     0�@     ��@     L�@     $�@     �@     ޣ@     Υ@     ʨ@     ��@     ��@     ��@     ��@     ĳ@     ��@     ��@     ��@     {�@     �@     U�@    �1�@    ���@    ���@    �5�@     G�@    ���@     s�@    ���@    @��@    �W�@     ��@    ��@    @��@    ���@    �m�@    ��@    �
�@    @|�@    ��@     D�@    �:�@    �z�@     ��@    ���@    �&�@     ��@     �@     Y�@     ��@     s�@     
�@     X�@     `�@     �g@     �J@      "@      �?        
�
fc1_b*�	    `C�   ��L?      x@! @کGy�?)�����?2��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`���(��澢f���侙ѩ�-߾E��a�Wܾ�iD*L�پ豪}0ڰ���������?�ګ�        �-���q=���?�ګ>����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�������:�              @      @      $@      @      $@      (@      �?      @       @      �?      "@      @      @      @       @              �?      @      �?      @      @       @      �?      �?      @      @      �?      @       @      �?      @              �?      �?      �?              �?               @              �?      �?              �?      �?              @              �?              �?              �?              �?               @              �?              �?       @              �?              �?      @      �?              �?      @      �?      @      �?      @      @      @       @      @      @      @      @      @      @       @      @      @      @      $@      @      @      .@      *@      *@      .@      7@      .@      ,@      @      @      �?       @        
�
fc1_out*�   ��5�?     pA!��b��@)@f����@2�        �-���q=豪}0ڰ>��n����>�u`P+d�>�[�=�k�>��~���>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�            @�A              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?       @      �?              @      @      @      �?      �?      @      @      @       @       @      "@      @      @      @      @       @      @      @      ,@      0@      (@      "@      *@      *@      1@      ;@      <@      :@      >@      A@      8@      :@      A@      D@      I@      D@      P@     @Q@     �T@     �Q@     @V@     @V@      Y@     �Y@      `@     �^@     �b@     �c@     `f@     `h@     `i@      j@     �n@     �p@     �s@     Pt@      v@     �z@      |@      }@     ��@     @�@      �@     ؆@     h�@     p�@     ��@     ��@     $�@      �@     ԕ@     ��@     X�@     ��@     ��@     �@      �@     T�@     Ҧ@     ��@     ��@     ȭ@     c�@     �@     �@     [�@     ��@     ��@     7�@     ��@    �Q�@     ��@     �@    ���@     7�@    ���@    ��@     ��@    �K�@     ��@    ���@     �@     ��@     
�@     ��@     Ĭ@     t�@     ��@     �@     ��@      x@     �b@     �Q@      A@      �?        
�
fc2_weights*�	    �r�   �y�r?      �@!szk���?)���G���?2�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������X$�z��
�}�������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�E'�/��x��i����v�6NK��2>�so쩾4>�����0c>cR�k�e>ڿ�ɓ�i>=�.^ol>�H5�8�t>�i����v>E'�/��x>f^��`{>K���7�>u��6
�>��ӤP��>�
�%W�>���m!#�>�4[_>��>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              @      (@      >@     �L@     �c@     pq@     `z@     (�@     ��@     ��@     ��@     ؕ@     ��@     ��@     ��@     0�@     T�@     X�@     T�@     �@     �@     �@     ��@     ��@     $�@     ��@     ��@     8�@     0�@     �@     ��@     �@     ��@     0@     p{@     �x@     Pv@     �t@      s@     �p@     `o@     @l@     `j@      g@     `e@     �d@      c@     @a@     �[@     �]@     �V@     �R@     �P@      T@     �Q@     �P@      M@     �I@      E@     �I@      M@     �@@      <@     �B@      9@      :@      5@      2@      ;@      .@      ,@      "@      .@      (@      @      (@      "@      "@       @       @      @      @       @      @      �?       @      @      @       @      @       @       @      @       @      �?       @      @              �?              �?      �?       @      �?      �?              �?       @               @              �?              �?               @              �?      �?      �?              �?              �?      �?      �?              �?      �?      �?       @               @      �?      @      �?       @       @      �?      @      @      @      @      @      @      @      @       @      @      @      "@      &@      $@      ,@      &@      ,@      6@      .@      8@      8@      5@      ?@      <@      <@      ;@      B@      E@      J@      D@      I@     �H@     @Q@      R@     @R@      S@      S@      U@      ]@     ``@     �[@     �`@     �f@     �d@     �j@      k@      l@     q@      n@     �r@     pv@      v@     �y@     P}@     �@     p�@     0�@     ��@     `�@      �@     ��@     �@     �@     �@     ��@     �@      �@     ��@     ��@      �@     x�@     ��@     \�@     ��@     ��@     �@     X�@     d�@     �@     8�@     ��@     �z@      s@      f@     �T@      B@      ,@      @      �?        
�
fc2_b*�	   �)jF�   २M?      h@!   ��{;�) �1�#?2�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��ߊ4F��h���`�        �-���q=})�l a�>pz�w�7�>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?x?�x�?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�������:�              �?              @       @     �E@      @      @       @      @      @      @      @      �?       @      �?      �?              �?              @              �?              �?              �?      �?              �?       @      �?      @       @      @      �?       @      @      @      �?      @      @      @      @      @      @      @      "@      @      @       @      �?        
�
fc2_out*�    ��?     pA!��b���@)[q���s@2�        �-���q=.��fc��>39W$:��>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�            �j�@               @              �?       @       @      �?      @      �?       @      �?              @              �?              @       @      @       @      @      @      @      @      �?       @      @       @      @      @      &@      @      $@      $@      (@      &@      $@      7@      5@      ,@      (@      4@      :@      :@      2@      ;@      ?@     �H@     �A@     �D@     �F@     �J@     �N@     �S@     �Q@     @S@     �V@      X@     �X@      [@     �_@     �`@     �a@     �d@     �g@     �h@     �h@      k@      m@     Ps@     �q@     �v@     w@     @z@     �z@     �}@     x�@      �@      �@      �@     ؈@     h�@     <�@     ��@     H�@     ��@     ��@     �@     К@     ��@      �@     8�@     ��@     P�@     ��@     b�@     "�@     "�@     ��@     4�@     ��@     b�@     �@     ��@     �@     *�@     �@     ��@     �@     �@     �@     ��@     ��@     p�@     8�@     �{@     �o@     �`@     �L@     �B@      (@        
�
logits_weights*�	    y�o�   ��i?      �@!  ��b��?)�E��*a?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾;�"�qʾ
�/eq
ȾG&�$��5�"�g���0�6�/n�������>
�/eq
�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�������:�              �?       @      �?      @      @      $@      0@      4@      1@      >@     �C@      @@      F@     �A@     �J@      J@     �E@     �E@      B@     �A@     �A@      :@      @@      :@      9@      3@      7@      1@      0@      5@      1@       @      (@      ,@      1@      $@      $@      @      @      $@      @      @      @      �?              @       @      @       @       @       @       @      @       @      �?      �?       @      �?               @              �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?      �?       @              @      @      �?       @       @       @      �?       @      �?       @       @       @      @      @       @      @       @      $@      @      (@       @      &@       @      &@       @      6@      &@      .@      ,@      4@      4@      :@      0@      E@      =@      B@     �A@     �F@      A@     �B@     �A@     �I@      D@      E@      J@     �G@     �D@      F@      A@      5@      9@      (@      "@       @       @        
�
logits_b*�	   ��)C�   `0�G?      $@!    ��0?)��=�Ǽ�>2��T���C��!�A�uܬ�@8���%�V6���bȬ�0���VlQ.�0�6�/n�>5�"�g��>>h�'�?x?�x�?+A�F�&?I�I�)�(?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?�������:�               @              �?              �?              �?              �?              �?              �?              �?              �?        
�

logits_out*�	    _�W�   @g�`?     ��@! ��[^��?)Р�-��s?2���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����������?�ګ��5�L�����]������|�~���MZ��K���u��gr��39W$:���.��fc���X$�z�������~�f^��`{������0c>cR�k�e>ڿ�ɓ�i>=�.^ol>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?�������:�              @      ,@     �F@     �`@     �r@     `z@      w@     r@     �n@     @k@     �r@     `p@     `b@     �a@      Z@     @[@     �Y@     �R@      O@      Q@     �Q@     �P@      Q@     �R@      E@     �L@      H@      G@      G@     �C@     �@@      <@      =@      >@      :@      7@      6@      :@      6@      2@      ,@      &@      4@      $@      &@      4@      "@      "@      .@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @              �?      �?              @              �?       @       @       @      @              �?              �?      �?               @               @      �?              �?              �?              �?              �?       @       @              �?              �?              �?              @      �?               @       @       @       @       @      @      @      @       @      @      @      @      @      @      @      @      @      @       @      &@      *@      *@      5@      ,@      8@      4@      9@      8@      C@      6@      A@     �A@     �I@     �H@     @P@     �P@     �T@     @T@     �U@     @U@     �X@     �V@     @R@     �R@     �T@     �Q@     �K@     �J@      S@      R@     �O@     �S@     �U@     �R@     �W@     �[@     �g@     @l@     �g@     �d@     `f@     �d@     �g@     �d@      d@      p@      x@      y@     �q@     �f@      `@     �O@      3@       @        

cross-entropy loss[@

accuracy��!>�:@e��      FoO�	���]u�A*��
�
conv1_weights*�	   ��^s�   �>�s?     ��@! �	E���)YZ�h�z�?2�
hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;����ž�XQ�þG&�$��5�"�g���=�.^ol>w`f���n>X$�z�>.��fc��>���]���>�5�L�>;9��R�>���?�ګ>��n����>�u`P+d�>�[�=�k�>��~���>K+�E���>jqs&\��>��~]�[�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�
              @      @      9@      G@     @S@     �Z@     �b@     �d@     �e@      f@     �c@     @g@     `e@     @d@     �a@     @Y@     @`@     �Z@     �U@      N@     �K@      M@      L@     �I@      B@     �D@     �@@      =@      >@      8@      =@      <@      0@      &@      *@      2@      "@      @      @      $@       @      @      @      @      @      @      @      @      @      @      �?              �?               @       @       @      �?       @      �?      �?       @              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @       @              �?      �?      �?              @      @       @      @      �?              @       @      @      �?      @      �?      @      @      @      @       @      @      @       @      (@       @      .@      @      *@      .@      0@      ,@      0@      5@      4@      4@      (@      ?@      ;@     �D@     �I@     �B@      K@      L@      O@      L@     �K@      Q@      T@     @Q@     �T@     @X@      Y@     �Y@     @Z@      Z@      S@     @V@     �V@      H@     �E@      9@      2@      @      �?        
�
conv1_b*�	    �fE?   @yO?      P@!  �i�?)J��?2@�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�������:@              �?      @      5@      0@      6@      �?        
�
	conv1_out*�   ���0@     @�A!�F�أy�A)t�C�A2�        �-���q=�H5�8�t>�i����v>�4[_>��>
�}���>�5�L�>;9��R�>��n����>�u`P+d�>0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@�������:�           �i�~A              �?              �?              �?               @              @              �?              �?      �?       @       @               @       @      @      @       @      @      $@      @      @      @      @       @      @      @      (@      @      $@      &@      0@      "@      .@      *@      0@      2@      0@      6@     �B@      7@      =@      @@      ?@     �F@      J@     �L@     �H@     �L@      O@      N@     �P@      S@      \@     �Y@     ``@      `@      d@     �c@     �d@     �h@      k@     �l@     @n@     p@     0s@      u@     �v@     @x@     �z@     `@     ��@     ��@     ��@     �@     P�@     ��@     �@     `�@     ��@     �@     �@     h�@     �@     4�@     ��@     n�@     �@     �@     ��@     �@     ��@     x�@     S�@     �@     ݴ@     ��@     &�@     z�@     R�@    ���@    ���@    ��@     �@     ��@    ���@    �J�@     @�@    ���@     ��@    ���@    ���@    @��@    ���@     i�@    �w�@    �>�@     �@    `�@    �]�@    ���@    ���@    `��@    ���@    0��@    ���@    ���@    p��@    �3�@    �� A    �OA    � A    xA    0KA     �
A     ~A    �!A    x�A    �|A     eA    �lA    ��A    ,@A    ��A    N� A    6u"A    �$A    4�%A    �'A    v�)A    ��+A    b-.A    �D0A    �1A    �2A    �3A    �5A    6A    W�6A    �s7A    ��7A    N7A    ��6A    �5A    �83A    ��0A    �4+A    �s$A    XBA    ��A    �A    �:�@     ��@     T�@        
�
conv1_maxpool*�   ���0@     @oA!��L��A)�Q(��s�A2�
        �-���q=�H5�8�t>�i����v>�4[_>��>
�}���>�����>
�/eq
�>;�"�q�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@�������:�
           �}�XA              �?              �?              �?       @              �?       @      �?              @              �?      �?       @              @              @       @      @      @      @               @      �?      @      @       @      @      ,@      @      @      @      @      2@      (@      (@      *@      "@      .@      "@      (@      2@      ?@      3@      5@      A@     �A@      >@     �I@     �E@     �I@     �E@     �P@     �P@     �R@     �T@     �X@     �Z@      V@      `@     ``@     @a@     @c@     �g@     `i@     `j@      i@     0q@     �q@      s@     �t@     @v@      x@      |@     ��@     ��@     @�@      �@     ��@     �@     �@     �@     ��@     <�@     p�@     Е@     x�@     ��@     ��@     $�@     Z�@     ܢ@     H�@     R�@     ��@     0�@     ܯ@     H�@     ��@     ܴ@     ��@     L�@     �@     ־@    �M�@    �{�@     ��@    �u�@     ��@    �m�@     �@    �x�@     @�@    �%�@    @��@    @)�@    ���@    �f�@    �T�@    ���@    �p�@    ��@     ��@    ���@     ��@    �m�@    0f�@    �5�@    @�@    P#�@     ��@    �B�@    ���@    0� A    ��A    PZA    �IA    �PA    ��
A    �A    �xA    0.A    H�A    l=A    T�A    �A    d<A    �A    8zA    @�A    �/A    �{A    �A    �A    �A    �A    �A    xc	A    � A    А�@    ��@     ��@     �@        
�
conv2_weights*�	   ��!t�   �{=v?      �@!(�93�|J@)����8��?2�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��K���7��[#=�؏���i����v��H5�8�t�w`f���n�=�.^ol�ڿ�ɓ�i�������M�6��>?�J�ہkVl�p>BvŐ�r>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?      @       @      6@      G@      _@      h@     �t@     P}@     ��@     h�@     4�@     �@     �@     H�@     ��@     ܖ@     <�@     h�@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ��@     @�@     H�@      �@     �@     `�@     Є@     x�@     (�@     `~@     �z@     `y@     �w@     u@      r@      s@      o@     �k@     @j@     �g@      e@     �d@      a@     �\@     �^@     �U@     �Z@     �V@     �W@     �O@     �O@     �I@     @Q@     �F@      H@     �H@     �D@      >@     �C@      <@      :@      (@      6@      4@      3@      2@      .@      $@      @      1@       @      @       @      &@      @       @      @      @      @      @      @      @      @      @      @      @      @      �?      @              �?      �?      �?      @      �?      @      �?               @      �?      �?       @              �?              �?              �?      �?              �?               @              �?              �?       @      �?      �?               @      �?      @      �?      �?              �?      �?      @       @      @      @      �?      @      �?      @      @       @      @       @      @      @      @      @      @      (@      &@      &@      3@      ,@      0@      2@      >@      6@      :@      4@      =@     �B@      >@      ?@     �F@     �F@     �L@      N@     �R@     �P@      T@     @X@     �T@     @]@     �[@     `b@     �a@     �c@     �e@     @e@     `i@     �l@     �p@     �n@      s@     Pu@     0w@     �w@     �{@     0~@     �@      �@     P�@     �@     ��@     ��@     Ў@      �@     �@     ��@     <�@     ��@     �@     p�@     �@     Ȣ@     ��@     ĥ@     �@     \�@     j�@     ��@     ��@     r�@     �@     �@     N�@     ��@     О@     �@     ��@     �~@     �m@     @W@      :@      @      �?        
�
conv2_b*�	     �    �YU?      P@!   s�-�?)�V��?2��S�F !�ji6�9����82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�������:�              �?              �?               @       @      �?       @               @      @      @      @      (@      5@      @      �?        
�
	conv2_out*�   @S�2@     @oA!cAy�?HA)�ʇN!
�A2�        �-���q=�i����v>E'�/��x>T�L<�>��z!�?�>���?�ګ>����>豪}0ڰ>��n����>�*��ڽ>�[�=�k�>��~���>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@�������:�           �ܸOA              �?              �?              �?              �?              @      �?               @              �?      �?       @      �?      @      �?      @       @       @      @      @      @      @      @       @      @      @      @       @      @      (@      @      "@      4@      .@      @      @      0@      $@     �A@      6@      9@      ?@      @@      <@      B@     �E@      >@     �E@     �B@     @Q@     @P@     �V@     �T@     �O@      X@     @W@      \@     �`@     @_@      c@     @f@     �g@     @g@     `f@     0p@     �o@     s@     �r@     �u@     �x@     �{@     �~@     p}@     �@     p�@     ��@     ��@     x�@     �@     0�@     X�@     ��@     ��@     P�@     (�@     P�@     ��@     ��@     t�@     *�@     ��@     ��@     �@     �@     �@     ��@     �@     \�@     ��@     ��@     ��@     �@     P�@     ��@     ��@    �I�@     B�@    ��@     ��@    ���@    ���@    @�@     A�@    �v�@    @��@    �_�@    �+�@     ��@    `r�@    �x�@    ���@    ���@     ��@    ���@    �i�@    ���@    `_�@    �E�@    ���@    )�@    �B�@    d�@    0��@    p[A    X�A    HqA    �A    ��A    p�	A    �A    ��A    �A    8DA    ��A    ��A    �0A    ��A    ,�A     /A    �yA    t�A    l�A    4xA    DA    �'A    8�A    \�A    ��A    p�A    �OA    ��A    p�A    `�A    �hA    � A     }�@    �e�@     ��@     �@     ��@     ,�@     ��@     @T@        
�
conv2_maxpool*�   @S�2@     @OA!���)aeA)Ό:�0�A2�
        �-���q=�*��ڽ>�[�=�k�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@�������:�
            �&A              �?              �?              �?      @              �?      �?      �?      @              �?      �?       @      @      @      �?      @      @      @               @      @      @      @      "@      @      @      @      @      &@      &@      @      @      "@      "@      7@      5@      7@      .@      2@      1@      9@      :@      >@      5@     �A@     �J@     �F@      D@      I@      L@     �P@     �L@     �S@     �U@     �V@     @Y@     �\@      `@     �b@     �b@     @c@     �d@     �i@     @g@     @j@      n@     �r@     �r@     Ps@     �v@     �z@      {@     @|@      �@     ��@      �@     �@     ��@     ؊@     ��@     ��@     đ@     p�@     t�@     ��@     ��@     �@     "�@     2�@     ��@     �@     ��@     n�@     �@     ��@     1�@     ˲@     ��@     5�@     X�@     ��@     �@    �-�@    ��@     ��@    ���@     ��@     C�@     l�@    ��@     ��@     ��@    @��@     n�@    �v�@    ���@    �g�@    ���@    ���@    ���@    @;�@    @��@    @��@    �$�@    `y�@    ���@     ��@    P�@    ���@    ���@    `.�@    ���@    ���@    �]�@    @�@    `��@    ���@    p� A    h|A    ��A    �A    ��A    0pA    � A    ���@    @=�@    `Q�@      �@    ���@    `��@    ���@     ��@     N�@     �@     �@     ��@     �X@        
�%
fc1_weights*�%	    o�u�   `�ow?      8A!�� 3�z@)P��p� @2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�6NK��2�_"s�$1�7'_��+/��'v�V,��`���nx6�X� ���-�z�!>4�e|�Z#>6NK��2>�so쩾4>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?       @      (@     �N@      l@     ��@     �@     $�@     ��@     ��@    ���@    ���@    @��@    @�@     ��@     ��@     /�@    ��@    �n�@    �R�@    ���@    �%�@    ���@     ��@    �[�@    � �@    ���@     X�@    @�@    @�@    ���@    � �@     ��@    ���@     l�@    �@�@     o�@     ��@     )�@     p�@     8�@     ޵@     ��@     d�@     ϰ@     >�@     $�@     ܨ@     "�@     ��@     �@     ��@     �@     0�@     ��@     p�@     t�@     x�@     ��@     ��@     �@     Ȍ@     ȉ@     0�@     ��@     ��@     (�@     ��@      {@     z@      v@     �u@     �s@     �q@     `p@      l@     @j@      f@     �g@      `@     �a@     �]@     �_@     @_@      Y@     @X@     @T@     @Q@     @Q@      H@      Q@     �E@      @@      L@     �C@      9@      8@      @@      <@      8@      1@      =@      5@      "@      .@      2@      &@      (@      @      "@      @      "@      @      $@      @      @      @      @      @      @      @      @       @       @       @       @      �?      �?      �?               @              �?              �?      �?      �?               @      �?      �?      @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?      �?               @       @       @      �?       @      @       @      @       @       @      @      @      @      @      "@      �?      "@      @      @       @       @       @      $@      $@      .@      4@      &@      1@      9@      5@      :@      A@     �@@      <@      C@     �G@     �I@      H@     @P@      G@      M@     @S@      S@     @V@     �T@     �]@     @]@     @^@     �]@     @c@     �d@     �g@     `i@     �g@     �m@     �n@     pq@     �t@      u@     @w@     �v@     P|@     0~@     p�@     ��@     ��@     @�@     X�@     @�@     �@     �@     �@     t�@     ��@     ��@     �@     ��@     z�@     ��@      �@     ��@     .�@     ک@     R�@      �@     �@     I�@     B�@     K�@     Թ@     ʻ@     5�@    �B�@     
�@     ��@    ���@    �R�@     ��@    �	�@    ���@    @n�@    @\�@    @9�@    �<�@     :�@    ���@    �=�@    ��@    ��@    `�@    �A�@     (�@    @��@    �C�@    `:�@     v�@    �e�@    `��@    �S�@    @�@    @��@    ��@    ���@     �@     0�@     �@     �z@     �Y@      3@      @        
�

fc1_b*�
	   @��I�   ��[?      x@! ����!�?)�*��'.?2��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v����(��澢f�����uE���⾮��%ᾙѩ�-߾
�/eq
Ⱦ����ž        �-���q=��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�������:�              @      @      @      "@      (@      @      @      �?      @       @      @      @      @      @       @       @      @       @      �?       @              �?               @      �?      �?              @               @      �?               @              �?              �?              �?      �?              �?              @              �?              �?              �?              �?              @       @               @       @      �?      �?      �?      @      @      �?      @       @      (@      &@      @      &@      *@      3@      :@      0@      8@     �D@      3@      6@      ,@      @       @        
�
fc1_out*�   ��7@     pA!��Ǎ['4A)=���� cA2�        �-���q=})�l a�>pz�w�7�>��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@�������:�            p��@              �?              �?              �?               @      �?      �?      �?      �?              �?              �?              �?      �?               @       @              �?       @       @      @       @      �?      �?               @      �?      @       @      @              @       @      @      @       @      @      @      @      &@       @       @      .@      .@      2@      8@      1@      6@      0@      <@      9@      8@      B@     �A@      <@      B@      G@     �H@     @P@     �S@     �P@     �P@     �W@      W@      Z@     �]@      _@     �a@     �b@      d@     �d@      i@     @l@     �h@     @n@     r@     `u@     �t@     �w@     �z@     �~@     p@     h�@     @�@     �@      �@     ��@     @�@     ��@     (�@     X�@      �@     @�@     �@      �@     ��@     ��@     P�@     ҧ@     6�@     ��@     ��@     +�@     ز@     �@     ��@     ��@     ��@     5�@     ��@    ���@     ��@    ���@     ��@     I�@    �^�@     ��@     6�@    ���@     ��@    ��@     ��@     �@     ±@     P�@     T�@     P�@     @|@     `c@      <@      �?        
�
fc2_weights*�	   ���q�   ��9v?      �@!nfz@)��8"ٌ�?2�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��[#=�؏�������~�6NK��2>�so쩾4>4�j�6Z>��u}��\>ہkVl�p>BvŐ�r>�i����v>E'�/��x>K���7�>u��6
�>T�L<�>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              @      9@      I@      ]@     �i@     �s@     `�@      �@     ��@     ��@     ��@     ��@     ؗ@     ��@     T�@     ��@     ��@     |�@     h�@     ��@     `�@     ,�@     ��@     X�@     ��@     �@     ��@     ��@     (�@      �@     x�@     �@     0~@     Px@     `y@     Pu@     t@     pu@     0q@     `o@     �j@     �g@     �e@      i@     �b@     �b@     �^@     �]@     �X@     @X@     �W@     �T@     �R@     �Q@      M@      N@     �F@      J@      D@     �D@      D@     �@@      A@      =@      4@      :@      6@      2@      .@      1@      $@      (@       @      &@      @      @       @      @      @      @      @       @      @      @      �?      @      @      @      �?       @      �?       @              �?               @               @      �?      �?      �?       @              �?      @      �?      �?              �?               @              �?              �?              �?              �?       @              �?      �?       @      �?              �?               @      �?              @      �?       @       @      @      @      @       @      @       @       @      @      @      @      @      "@      &@       @       @      @      ,@      "@      2@      7@      (@      6@      @@      A@      8@      :@      @@     �A@     �E@      O@     �G@      K@     �L@     �P@     �Q@     �N@     @Q@     �Y@     �X@     �`@     �\@     �b@     �d@     �b@     @h@     �j@     �g@      p@     @r@     �r@     u@      w@     �z@     0|@      �@     h�@     `�@     P�@     @�@     ��@     `�@     �@     h�@     �@     T�@     ��@     X�@     \�@     ��@     D�@     ��@     ��@     @�@     d�@     l�@     L�@     `�@      �@     ��@     ��@      �@     �@     �z@     �m@     `a@      S@      8@       @      �?      �?        
�	
fc2_b*�		    lpJ�   �H�Y?      h@!  d���?)M�>��?2�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1��ߊ4F��h���`��uE���⾮��%�        �-���q=pz�w�7�>I��P=�>O�ʗ��>>�?�s��>1��a˲?6�]��?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�������:�              �?      &@      @      @      @      ?@      @      @      @       @      @      �?              �?      �?       @      �?              �?      �?      �?              �?              �?               @               @              �?              �?              �?              �?               @               @      �?      �?      @      @       @      �?       @      �?      @      @      @      �?      �?       @      @      @      @       @      @      @      "@      &@       @      "@      @      �?        
�
fc2_out*�   `�@     pA!��:C<-�@)����`�@2�        �-���q=���%�>�uE����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>>�?�s��>�FF�G ?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@�������:�            �l�@              �?              �?              �?               @              �?      �?              �?      �?      �?              �?       @      �?      @      @              @       @      @      @       @       @      @       @      @       @      @       @      @      @      @      @      @       @      .@      ,@      &@      1@      *@      5@      *@      6@      *@      4@      5@      7@      A@     �A@      F@      =@     �K@     �G@      K@      O@     �S@     �S@     �S@      W@     �X@     �V@     @]@      _@     �e@     �b@     �e@      j@      j@     �k@      o@     �n@     �q@     Pu@     �v@     @y@     p}@     �}@     P�@     �@     �@     `�@     `�@     ��@     `�@     ��@     �@     X�@     ��@     ��@     @�@     ��@     Ě@     p�@     ,�@     Ơ@     @�@     �@     Ԧ@     L�@     H�@     ~�@     ��@     �@     f�@     �@     F�@     �@     ̮@     n�@     F�@     N�@     ��@     ��@     �@     �@     ��@     8�@     �t@     �g@     @U@      @@       @        
�
logits_weights*�	    ��r�   �Spn?      �@!  H�]�?)h�m�joh?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(�E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ�*��ڽ�G&�$����ӤP�����z!�?��a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              �?      �?      �?      @      @      .@      1@      .@      ;@      :@     �C@      C@     �I@      >@      J@      C@     �I@     �I@      A@     �E@      <@      @@      ?@      <@      3@      0@      4@      3@      5@      6@      *@      ,@      &@      @      $@      (@      &@       @      @      @      "@      @      @      @      @      @      @       @      �?      @      �?      @               @       @      �?      @              �?      �?      �?      �?              �?              �?       @      �?              �?              �?              �?               @      �?       @              �?      @      @      �?       @      �?      @      @      @              �?              @      @      �?      �?      @      @      &@      "@      (@      &@      @      *@      (@      0@      0@      0@      .@      2@      1@      2@      1@     �A@      ?@      @@      8@      E@      E@      A@     �F@      L@      D@      B@      E@      E@      I@     �D@      ?@      C@      4@      .@      0@      $@      @       @        
�
logits_b*�	    N~K�   �lB?      $@!   ���6?)��f�K�>2�IcD���L��qU���I��T���C��!�A�ji6�9���.����ڋ��vV�R9���[�?1��a˲?+A�F�&?I�I�)�(?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?�������:�              �?              �?              �?              �?              �?              �?               @              �?              �?        
�

logits_out*�	    iIп   �W��?     ��@! �"B�oH@)�����E@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7���f�ʜ�7
������1��a˲���[��pz�w�7�>I��P=�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?�������:�              @      ,@      6@      ;@     �H@     @S@     @T@     �W@      ^@     @\@      ^@      `@     �^@     �^@     `f@     �f@     �l@     �n@      n@      p@      n@      l@     �e@     �a@     �b@      _@     @T@     @S@      Q@     �N@     �M@      I@      I@     �D@      H@     �B@      C@     �G@      D@      <@      =@      ?@      :@      0@      1@      0@      2@      @      *@      &@      $@      &@      @       @      @       @      @      @      @      @      @      @      @      @       @       @       @      �?               @              �?              �?              �?              �?               @      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?       @      �?       @              @       @       @      �?       @      �?              �?       @              @      @       @      �?      @      @      "@       @      @      @      $@      @      @      $@      6@      $@      &@      @      2@      8@      7@      0@      4@      6@      :@      >@      ?@     �C@      B@      E@     �E@     �K@     �L@     �U@      U@     @Y@     @[@      b@     `b@     �d@      i@     �l@      m@     p@     �p@     @r@     �r@     �q@     r@     �q@      p@     �k@     @i@     �e@     `c@      ^@      W@     �M@     �L@      @@      1@      $@      @       @        

cross-entropy loss��@

accuracy��'>�#�B