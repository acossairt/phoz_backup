?	??Q??\@??Q??\@!??Q??\@	??M?B????M?B??!??M?B??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q??\@lˀ??"F@Aȴ6???Q@Y??Tka??*	#??~j?e@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeats?4?B??!?5bkpr;@)?%??p??1??~??7@:Preprocessing2F
Iterator::ModelDj??4ӱ?!;??(?*D@)ؚ?????1??Tt??4@:Preprocessing2U
Iterator::Model::ParallelMapV2?9τ&??!?o??,g3@)?9τ&??1?o??,g3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?Z_??!	??mBO9@)?-=??ɜ?1>D?H0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice* ?3h???!?yw͠"@)* ?3h???1?yw͠"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??1>?^??!?`e?t?M@)?=^H????1?5??M?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ng_y?~?!Nd?KJ@)?ng_y?~?1Nd?KJ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??u?X??!"~?ो;@);?f??o?1? ??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??M?B??I??<Xo?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	lˀ??"F@lˀ??"F@!lˀ??"F@      ??!       "      ??!       *      ??!       2	ȴ6???Q@ȴ6???Q@!ȴ6???Q@:      ??!       B      ??!       J	??Tka????Tka??!??Tka??R      ??!       Z	??Tka????Tka??!??Tka??b      ??!       JCPU_ONLYY??M?B??b q??<Xo?X@Y      Y@q?=}It?F@"?	
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?45.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 