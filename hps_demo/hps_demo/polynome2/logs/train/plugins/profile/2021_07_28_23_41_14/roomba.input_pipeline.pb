	o???w?@o???w?@!o???w?@	G?#????G?#????!G?#????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$o???w?@??-?R??A?V'gH-?@Y?L???"@*	T㥛<[?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatez?Cn??@!OhB??F@)??j???@1?$??F@:Preprocessing2F
Iterator::Modelb֋???!@!?fwj"?J@)?);??.@1?????@@:Preprocessing2U
Iterator::Model::ParallelMapV2F??0E@!?E'?+Z4@)F??0E@1?E'?+Z4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvŌ?? ??!?f0?y??)?;?????1З?$B??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?C?l????!??K6????)?C?l????1??K6????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip
F??-@!]????DG@)?l??爜?1??P??K??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3?z????!U;Q?%??)3?z????1U;Q?%??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?1????@!??v?z?F@)?6?~t?1	??a???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9G?#????I?k????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-?R????-?R??!??-?R??      ??!       "      ??!       *      ??!       2	?V'gH-?@?V'gH-?@!?V'gH-?@:      ??!       B      ??!       J	?L???"@?L???"@!?L???"@R      ??!       Z	?L???"@?L???"@!?L???"@b      ??!       JCPU_ONLYYG?#????b q?k????X@