	
?O?x?a@
?O?x?a@!
?O?x?a@	??X?|Ǹ???X?|Ǹ?!??X?|Ǹ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$
?O?x?a@a?riCV@A?%䃞?I@Y?H?F?q??*??v??nh@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatˡE?????!??g
w?<@)?y?'L??15???G8@:Preprocessing2U
Iterator::Model::ParallelMapV2?IEc????!b#???4@)?IEc????1b#???4@:Preprocessing2F
Iterator::Model?n?KS??!???x? D@)?i4???1*c@?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(CUL????!$??F??8@)F??(&o??12???	l0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??G?`??!?mǷ?] @)??G?`??1?mǷ?] @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor
?Y2ǂ?!???@)
?Y2ǂ?1???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???%??!hN?v?M@)vöE???1`6ԗ.
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?<*??!t??^??:@) ?O??n?1??w????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 63.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??X?|Ǹ?I??? ??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	a?riCV@a?riCV@!a?riCV@      ??!       "      ??!       *      ??!       2	?%䃞?I@?%䃞?I@!?%䃞?I@:      ??!       B      ??!       J	?H?F?q???H?F?q??!?H?F?q??R      ??!       Z	?H?F?q???H?F?q??!?H?F?q??b      ??!       JCPU_ONLYY??X?|Ǹ?b q??? ??X@