	_^?}??m@_^?}??m@!_^?}??m@	`?2?J???`?2?J???!`?2?J???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$_^?}??m@l??g?@A+*???l@YN??????*	??(\??n@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????M???!՜7?f:@)???o^???1~?Q?Lo3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??1zn??!u????@@)xC8٦?1??;?2@:Preprocessing2F
Iterator::Model?'c|????!?????`@@)??7h???1L3??1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??"M???!???C?/@)??"M???1???C?/@:Preprocessing2U
Iterator::Model::ParallelMapV2ɑ???ˢ?!ŧ????-@)ɑ???ˢ?1ŧ????-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??+?z???![??F.?@)??+?z???1[??F.?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipOWw,?I??! ????P@)?C?R???1\)?j7?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz?(???!k?X?A@)??C???r?1?^? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9`?2?J???I????I?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l??g?@l??g?@!l??g?@      ??!       "      ??!       *      ??!       2	+*???l@+*???l@!+*???l@:      ??!       B      ??!       J	N??????N??????!N??????R      ??!       Z	N??????N??????!N??????b      ??!       JCPU_ONLYY`?2?J???b q????I?X@