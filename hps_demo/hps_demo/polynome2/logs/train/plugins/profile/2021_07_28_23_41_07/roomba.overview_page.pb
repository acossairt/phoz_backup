?	l????v?@l????v?@!l????v?@	?Tn?c???Tn?c??!?Tn?c??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$l????v?@˅ʿ?W??AH3M???@Y?=?4a?@*	rh??r?@2U
Iterator::Model::ParallelMapV2+hZbe?@!T?Bf?pD@)+hZbe?@1T?Bf?pD@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate??@!h?`hC@)B??=?@1??7?46C@:Preprocessing2F
Iterator::Model??? 4?@!?N[??6N@)?H?<?+@15?0??3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatū?m?ǭ?!??V?[???)????˚??1}4????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceQ?l???!????????)Q?l???1????????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4?\?a@!??i?C@)?????ٕ?1巃?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?t??????!9(?1i???)?t??????19(?1i???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?5Φ#@!??YٻlC@)??e?O7p?1?MX?u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?Tn?c??I?W#=8?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	˅ʿ?W??˅ʿ?W??!˅ʿ?W??      ??!       "      ??!       *      ??!       2	H3M???@H3M???@!H3M???@:      ??!       B      ??!       J	?=?4a?@?=?4a?@!?=?4a?@R      ??!       Z	?=?4a?@?=?4a?@!?=?4a?@b      ??!       JCPU_ONLYY?Tn?c??b q?W#=8?X@Y      Y@q6-2???y?"?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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