	VE?ɨ?>@VE?ɨ?>@!VE?ɨ?>@	 )a?9??? )a?9???! )a?9???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$VE?ɨ?>@S"?^?@A???6??6@Y?x?JxB??*	A`??"??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?~?7???!Q/??z?P@)0+?~N??1??g??N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatsL????!Bpj[	?)@)qqTn????1?W^X%@:Preprocessing2U
Iterator::Model::ParallelMapV2s???6???!?Cj?MS"@)s???6???1?Cj?MS"@:Preprocessing2F
Iterator::Model@?իȸ?!7r???1@)??0? ??1FAy???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceX<?H?ۚ?!????
?@)X<?H?ۚ?1????
?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?J?h??!s#???T@)??q?d???1gꟊ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorН`?un??!??K??>@)Н`?un??1??K??>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV?@?)V??!?ت??P@)?G??|v?1FcӍw??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 25.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 )a?9???I?=}?u?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S"?^?@S"?^?@!S"?^?@      ??!       "      ??!       *      ??!       2	???6??6@???6??6@!???6??6@:      ??!       B      ??!       J	?x?JxB???x?JxB??!?x?JxB??R      ??!       Z	?x?JxB???x?JxB??!?x?JxB??b      ??!       JCPU_ONLYY )a?9???b q?=}?u?X@