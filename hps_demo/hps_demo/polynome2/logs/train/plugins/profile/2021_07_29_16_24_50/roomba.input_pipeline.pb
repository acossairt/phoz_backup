	?\m??F]@?\m??F]@!?\m??F]@	.?Ë+??.?Ë+??!.?Ë+??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?\m??F]@?	????A?gB???\@YI???M??*	?G?z?@2U
Iterator::Model::ParallelMapV2?T2 T???!??YIH@)?T2 T???1??YIH@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate>{.S????!-o???H@)????	??1??????G@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatz?΅?^??!.??pQ???)??????1?9?!>??:Preprocessing2F
Iterator::Model?v?ӂW??!????H@)xADj?Ť?1Kd'?ɽ??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceh??`ob??!S?M"????)h??`ob??1S?M"????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!N8=hUI@)?[?~l???1??{??{??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorh>?n?K??!b~??h??)h>?n?K??1b~??h??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?B?i????!nsR?HOH@)<??~Kq?1⥌?E4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9.?Ë+??I????S?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	?????	????!?	????      ??!       "      ??!       *      ??!       2	?gB???\@?gB???\@!?gB???\@:      ??!       B      ??!       J	I???M??I???M??!I???M??R      ??!       Z	I???M??I???M??!I???M??b      ??!       JCPU_ONLYY.?Ë+??b q????S?X@