?	ѱ?J??g@ѱ?J??g@!ѱ?J??g@	h?O+?-??h?O+?-??!h?O+?-??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ѱ?J??g@W?o?\@A????R@Y?"?k$??*	?V?
?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???ȭ?@!?{????W@)?H????@1%\hҺW@:Preprocessing2U
Iterator::Model::ParallelMapV2a?^Cp\??!h?GEB???)a?^Cp\??1h?GEB???:Preprocessing2F
Iterator::Model????ƽ?!???T@)???~31??1?5ä????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?H?F?q??!??????)?<0???1??_????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?J>v(??!-??/???)?J>v(??1-??/???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??"h?@!??WH_UX@)m?????1???!????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensork?) ƃ?!?x?)?S??)k?) ƃ?1?x?)?S??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?x??n?@!$???W@)ђ???w?1n#(L|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9h?O+?-??I	,u?t?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W?o?\@W?o?\@!W?o?\@      ??!       "      ??!       *      ??!       2	????R@????R@!????R@:      ??!       B      ??!       J	?"?k$???"?k$??!?"?k$??R      ??!       Z	?"?k$???"?k$??!?"?k$??b      ??!       JCPU_ONLYYh?O+?-??b q	,u?t?X@Y      Y@q?h?{?+@"?
both?Your program is POTENTIALLY input-bound because 60.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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