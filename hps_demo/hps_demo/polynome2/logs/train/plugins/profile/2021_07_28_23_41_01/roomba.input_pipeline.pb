	?T[(?@?T[(?@!?T[(?@	????W??????W??!????W??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?T[(?@??[???@A?|?rٵ@Yr6??@*	?K7?a??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateL8??+@!X?
zdJ@)JΉ=??@1?c???-J@:Preprocessing2F
Iterator::Model?}??ř@!??[T0G@)/?>:u%
@1????>L@@:Preprocessing2U
Iterator::Model::ParallelMapV2?w?'-??!?ЗV?+@)?w?'-??1?ЗV?+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath?o}Xo??!4??r???)??,?Ŧ?1?
?N?b??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceÀ%W????!tW??d??)À%W????1tW??d??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0?k???@!@y褫?J@)???V	??1E]??????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorE?
)????!Խ?4@??)E?
)????1Խ?4@??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+?ެ1@!?{P?kJ@)z???x?18[n???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????W??I?O
jP?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??[???@??[???@!??[???@      ??!       "      ??!       *      ??!       2	?|?rٵ@?|?rٵ@!?|?rٵ@:      ??!       B      ??!       J	r6??@r6??@!r6??@R      ??!       Z	r6??@r6??@!r6??@b      ??!       JCPU_ONLYY????W??b q?O
jP?X@