	;9Cq??_@;9Cq??_@!;9Cq??_@	?	<mЈ???	<mЈ??!?	<mЈ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$;9Cq??_@~r 
&#@AOt	X]@Yx??#????*	??ʡE?@2U
Iterator::Model::ParallelMapV2؝?<????!????$?A@)؝?<????1????$?A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?{eު???!??G$??A@)???u6???1n?????@@:Preprocessing2F
Iterator::Model?⪲????!??`w'L@)?x?????1?5ؤ5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat|?????! ???ng@)???????1?{?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?E?n?1??![㡇e? @)?E?n?1??1[㡇e? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4??o???!?B???E@)%???????1s?"G???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???W?x?!z?ej??)???W?x?1z?ej??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? w?(??!?WgU3B@)??҈?}n?1f;+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?	<mЈ??I?Ò/w?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~r 
&#@~r 
&#@!~r 
&#@      ??!       "      ??!       *      ??!       2	Ot	X]@Ot	X]@!Ot	X]@:      ??!       B      ??!       J	x??#????x??#????!x??#????R      ??!       Z	x??#????x??#????!x??#????b      ??!       JCPU_ONLYY?	<mЈ??b q?Ò/w?X@