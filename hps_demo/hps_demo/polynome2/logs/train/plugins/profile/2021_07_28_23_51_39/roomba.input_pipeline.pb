	?j?0W^@?j?0W^@!?j?0W^@	?Pø?????Pø????!?Pø????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?j?0W^@W&?R?i@@Ad?? wV@YT?{F"4??*	&1?vh@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/?????!?9?&?>@)Q???Y??1y&???L9@:Preprocessing2U
Iterator::Model::ParallelMapV2??????!,?,?w2@)??????1,?,?w2@:Preprocessing2F
Iterator::Model?????T??!Sۋ?#LA@)ĵ??^(??1{?X?^ 0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice\???(\??!??:աL/@)\???(\??1??:աL/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$'?
b??!V:?YP@)I?2?喖?1DJ}&??&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?ߢ????!M?Y?\?@)y?ߢ????1M?Y?\?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW??,???!û??H?4@)?:?zj??1??$?_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=_?\6:??!-p?.7@)4GV~?q?1M?R=i?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Pø????I,?QS?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W&?R?i@@W&?R?i@@!W&?R?i@@      ??!       "      ??!       *      ??!       2	d?? wV@d?? wV@!d?? wV@:      ??!       B      ??!       J	T?{F"4??T?{F"4??!T?{F"4??R      ??!       Z	T?{F"4??T?{F"4??!T?{F"4??b      ??!       JCPU_ONLYY?Pø????b q,?QS?X@