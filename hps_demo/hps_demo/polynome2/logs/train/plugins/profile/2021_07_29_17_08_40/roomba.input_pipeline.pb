	od?Ve@od?Ve@!od?Ve@	+?~???+?~???!+?~???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$od?Ve@?+ٱ?%@Ax$(??c@Y&䃞ͪ??*	??? ?g@2U
Iterator::Model::ParallelMapV2????????!B?6?5@)????????1B?6?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat<FzQ???!N+??? 9@)???	????1%?D??5@:Preprocessing2F
Iterator::Model?kA??!??!;?ޥ?PE@)???"???15????4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>w??׹??!g53Z=+@)>w??׹??1g53Z=+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate`??Ù??!???NB?8@)?߆?y??1[??i*?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip
?2?&??!?!Z6?L@)?74e???1?{?N@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensors?ۄ{e~?!??%ɫ@)s?ۄ{e~?1??%ɫ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	???k??!?S?i??:@)?ZӼ?m?1?s??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9*?~???I?L  ??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?+ٱ?%@?+ٱ?%@!?+ٱ?%@      ??!       "      ??!       *      ??!       2	x$(??c@x$(??c@!x$(??c@:      ??!       B      ??!       J	&䃞ͪ??&䃞ͪ??!&䃞ͪ??R      ??!       Z	&䃞ͪ??&䃞ͪ??!&䃞ͪ??b      ??!       JCPU_ONLYY*?~???b q?L  ??X@