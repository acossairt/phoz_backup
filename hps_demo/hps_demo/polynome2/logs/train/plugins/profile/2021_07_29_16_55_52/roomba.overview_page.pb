?	-$`??u@-$`??u@!-$`??u@	??P??@??P??@!??P??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-$`??u@?u?X???A?dU?u@Y??nJ @*	??C?2?@2F
Iterator::Model???R?@!z,*i?L@)q??Ŷ@1??4??cD@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?G??5|@!0;Ëw:@)?G??5|@10;Ëw:@:Preprocessing2U
Iterator::Model::ParallelMapV2^??K@!`?g?k0@)^??K@1`?g?k0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ǚ??@!?p<s?E@)!撪??@1??{FBw/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW!?'?>??!h?=?s@??)?T??7??1LA?}???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7߈?Y?@!???ՖfE@)???9"ߕ?1?I7????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?1w-!??!q8?+???)?1w-!??1q8?+???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI.?!??@!???sE@)	]??u?1?Bu?u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??P??@I?Q&kX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?u?X????u?X???!?u?X???      ??!       "      ??!       *      ??!       2	?dU?u@?dU?u@!?dU?u@:      ??!       B      ??!       J	??nJ @??nJ @!??nJ @R      ??!       Z	??nJ @??nJ @!??nJ @b      ??!       JCPU_ONLYY??P??@b q?Q&kX@Y      Y@qw????y?"?
device?Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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