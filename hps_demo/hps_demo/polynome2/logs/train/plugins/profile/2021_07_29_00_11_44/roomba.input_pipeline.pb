	?????M@?????M@!?????M@	?Je??????Je?????!?Je?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????M@?74e?MK@A]???2e@Y?Q???T??*	???S?!h@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??$w?D??!ɷ?rK?>@)?Xİè?1!}L??9@:Preprocessing2F
Iterator::Model????"???!?????3D@)P?s'???1K????5@:Preprocessing2U
Iterator::Model::ParallelMapV2!sePmp??!?P?҆?2@)!sePmp??1?P?҆?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate5`??i??!3?#mjT5@)j?{?ԗ??1???Z?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice ?K????!W??z?$@) ?K????1W??z?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?#F?-t??!LB1?M@)<P?<???1q?骬S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?1˞??!???jF@)?1˞??1???jF@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy=???!????+d7@)??hUMp?1v???~ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 91.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Je?????I[M஄?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?74e?MK@?74e?MK@!?74e?MK@      ??!       "      ??!       *      ??!       2	]???2e@]???2e@!]???2e@:      ??!       B      ??!       J	?Q???T???Q???T??!?Q???T??R      ??!       Z	?Q???T???Q???T??!?Q???T??b      ??!       JCPU_ONLYY?Je?????b q[M஄?X@