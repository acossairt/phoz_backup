	?f???X@?f???X@!?f???X@	 ?????? ??????! ??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?f???X@???3K?$@A	Q???V@Y? ?X4???*	J+??l@2U
Iterator::Model::ParallelMapV2u???mn??!?/w?F?;@)u???mn??1?/w?F?;@:Preprocessing2F
Iterator::Model?L/1????!K?A!??G@) ??*Q???1???5?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatɮ???{??!?|?g??6@)D??????1?P??`2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??F????!J6_!y?7@)??*Q????1???G??/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???ꫫ??!4????@)???ꫫ??14????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?[?nK???!?u??ADJ@)r5?+-#??163??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor>[{??!I?p]?@)>[{??1I?p]?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?z????!b???9@)?Y??U?p?1$?B?p??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??????I
?w??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???3K?$@???3K?$@!???3K?$@      ??!       "      ??!       *      ??!       2		Q???V@	Q???V@!	Q???V@:      ??!       B      ??!       J	? ?X4???? ?X4???!? ?X4???R      ??!       Z	? ?X4???? ?X4???!? ?X4???b      ??!       JCPU_ONLYY??????b q
?w??X@