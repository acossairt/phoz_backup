?	l?V^?D@l?V^?D@!l?V^?D@	W??4???W??4???!W??4???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$l?V^?D@???#*
B@Aq??]?@Y)?{?i???*	?G?zi@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath??s???!??fY:@)??u?X???1??????5@:Preprocessing2U
Iterator::Model::ParallelMapV2K????/??!?C\?5@)K????/??1?C\?5@:Preprocessing2F
Iterator::Model??b('ڵ?!??69BE@)?V?????1(?1??4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateP?>?©?!1Ĭ??9@).s??/??1+??G4?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceD?Ac&??!???&? @)D?Ac&??1???&? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipA}˜.???!??ƽL@)?.l?V^??12?NJ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?s??q5??!7n?ض@)?s??q5??17n?ض@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\;Qi??!??y$?:@)? "5?bj?1x???x???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9W??4???I	R??!?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???#*
B@???#*
B@!???#*
B@      ??!       "      ??!       *      ??!       2	q??]?@q??]?@!q??]?@:      ??!       B      ??!       J	)?{?i???)?{?i???!)?{?i???R      ??!       Z	)?{?i???)?{?i???!)?{?i???b      ??!       JCPU_ONLYYW??4???b q	R??!?X@Y      Y@q?K??S@"?	
both?Your program is POTENTIALLY input-bound because 89.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?78.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 