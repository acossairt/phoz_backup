?	7?Fc@7?Fc@!7?Fc@	:KM&@:KM&@!:KM&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$7?Fc@W?'b@A???W?`@Y????11@*	???(\??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZ?????!!??q??V@)|Bv??&??1-/???V@:Preprocessing2U
Iterator::Model::ParallelMapV2??R{m??!?G@)??R{m??1?G@:Preprocessing2F
Iterator::Model??%?2???!Sd+???@)?L?T???1?	1Hi@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??y7??!??!?8=@)?<?;k???1??t>"? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??.4?i??!??!?0???)??.4?i??1??!?0???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipG??????!?I???W@)?,
?(z??1ŷH_<q??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?M?d???!&Ae-????)?M?d???1&Ae-????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo.2???!??V9W@)#??t?10??W???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9:KM&@I?8=?V6V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W?'b@W?'b@!W?'b@      ??!       "      ??!       *      ??!       2	???W?`@???W?`@!???W?`@:      ??!       B      ??!       J	????11@????11@!????11@R      ??!       Z	????11@????11@!????11@b      ??!       JCPU_ONLYY:KM&@b q?8=?V6V@Y      Y@q[g{?????"?
both?Your program is MODERATELY input-bound because 11.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
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