	???;?X@???;?X@!???;?X@	؝K??W??؝K??W??!؝K??W??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???;?X@?V????@@A;oc?#YP@Y??<?!7??*	?????5t@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Dh???!?)^"?E@)?e3????1g???'?B@:Preprocessing2U
Iterator::Model::ParallelMapV2d?w?W??!??R%?1@)d?w?W??1??R%?1@:Preprocessing2F
Iterator::Model'?%??s??!G??/A@)?b?J!???1?"????0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Y?rL??! ??L2@)???mR??1}?ߖ.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice??U?&??!8????W@)??U?&??18????W@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??S9?)??!]?y9hP@)?mR?X???1^Z?I??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???P???!?#?*$@)???P???1?#?*$@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9؝K??W??I1Z4T?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V????@@?V????@@!?V????@@      ??!       "      ??!       *      ??!       2	;oc?#YP@;oc?#YP@!;oc?#YP@:      ??!       B      ??!       J	??<?!7????<?!7??!??<?!7??R      ??!       Z	??<?!7????<?!7??!??<?!7??b      ??!       JCPU_ONLYY؝K??W??b q1Z4T?X@