?	#???yZ@#???yZ@!#???yZ@	??gh????gh??!??gh??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#???yZ@7??:r0C@AW?c#?P@Y??? 4???*	?I??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ʉvU@!?????T@)8,??6@1<zRLG?T@:Preprocessing2F
Iterator::Model~V?)????!И????-@)д??hd??1@?,??,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatY32?]???!Ђ?PfL??)?\?E??1z~???,??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??켍͞?!????????)??켍͞?1????????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip|??%?@!??,?IU@)?!7????1??7?Հ??:Preprocessing2U
Iterator::Model::ParallelMapV2?5?ڋh??!
,?????)?5?ڋh??1
,?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU?????!?z*????)U?????1?z*????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?QF\ Z@!ds<?l?T@)
???I'r?1?????0??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 36.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??gh??I???1/?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7??:r0C@7??:r0C@!7??:r0C@      ??!       "      ??!       *      ??!       2	W?c#?P@W?c#?P@!W?c#?P@:      ??!       B      ??!       J	??? 4?????? 4???!??? 4???R      ??!       Z	??? 4?????? 4???!??? 4???b      ??!       JCPU_ONLYY??gh??b q???1/?X@Y      Y@q??=]??"?
both?Your program is POTENTIALLY input-bound because 36.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 