	?J!?KQ@?J!?KQ@!?J!?KQ@	??Hw<?????Hw<???!??Hw<???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?J!?KQ@?~?Ϛ_@A?wcAaP@Y ~?{????*	-??燐l@2F
Iterator::Model
0,?-??!Ơ????D@)Ks+??X??1??߹#m6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?V?????!?9?5?\:@)q???h ??1r???(?3@:Preprocessing2U
Iterator::Model::ParallelMapV2??,z???!?G5*?2@)??,z???1?G5*?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?=%????!	>?;6@)??מY??1T9?F1\+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceW횐???!????<!@)W횐???1????<!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Q?H??!:_lYkM@)??ދ/ړ?1&?k??? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA?,_????!6"??#@)A?,_????16"??#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZ??c!:??!a????8@)??????p?1??ɚ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Hw<???I?[?a??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~?Ϛ_@?~?Ϛ_@!?~?Ϛ_@      ??!       "      ??!       *      ??!       2	?wcAaP@?wcAaP@!?wcAaP@:      ??!       B      ??!       J	 ~?{???? ~?{????! ~?{????R      ??!       Z	 ~?{???? ~?{????! ~?{????b      ??!       JCPU_ONLYY??Hw<???b q?[?a??X@