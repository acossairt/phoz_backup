	???l|N@???l|N@!???l|N@	6?n?<*??6?n?<*??!6?n?<*??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???l|N@X?|[?$.@A?????F@Y9a?hV???*	??S㥇m@2F
Iterator::Model???R?1??!9!	-C@)?f׽???1lDu?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?\5????!?Ynl?9@)Y???j??1΋c?V5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate????ֱ?!V?5=@)2V??W??1?????4@:Preprocessing2U
Iterator::Model::ParallelMapV22?Lڤ?!??{=1@)2?Lڤ?1??{=1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceUD? ??!vf"w!@)UD? ??1vf"w!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?? :??!a8+?U`@)?? :??1a8+?U`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4?;???!??????N@){??????1?}g?r@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J?(??!q3p?έ?@)G6uu?1?X?)?u@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no96?n?<*??IZ?/???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X?|[?$.@X?|[?$.@!X?|[?$.@      ??!       "      ??!       *      ??!       2	?????F@?????F@!?????F@:      ??!       B      ??!       J	9a?hV???9a?hV???!9a?hV???R      ??!       Z	9a?hV???9a?hV???!9a?hV???b      ??!       JCPU_ONLYY6?n?<*??b qZ?/???X@