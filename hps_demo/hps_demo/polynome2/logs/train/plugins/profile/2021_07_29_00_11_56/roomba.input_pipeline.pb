	>>!;o?@>>!;o?@!>>!;o?@	Z?3aa@Z?3aa@!Z?3aa@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$>>!;o?@????m3??A?fh<?@Y$???x??*	!?rh?t@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)&o?????!?PX?G@)wj.7???1j/?-ʗD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????켱?!Q?4x?5@)?2??(??1?D?:?1@:Preprocessing2U
Iterator::Model::ParallelMapV2:;%???!?]?w?+@):;%???1?]?w?+@:Preprocessing2F
Iterator::Model?J>v(??!% !K??9@)BZc?	???1??[??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0_^?}??!??7-??R@),I???p??1?$?*W@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??J
,??!??&g?@)??J
,??1??&g?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??>$D??!?????@)??>$D??1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapp?n?????!)??Z=?G@)????#*t?1|??
Ɍ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Z?3aa@I?aF??lX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????m3??????m3??!????m3??      ??!       "      ??!       *      ??!       2	?fh<?@?fh<?@!?fh<?@:      ??!       B      ??!       J	$???x??$???x??!$???x??R      ??!       Z	$???x??$???x??!$???x??b      ??!       JCPU_ONLYYZ?3aa@b q?aF??lX@