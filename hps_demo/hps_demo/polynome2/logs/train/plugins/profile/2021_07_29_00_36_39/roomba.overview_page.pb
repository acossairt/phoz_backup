?	"ĕ?7.@"ĕ?7.@!"ĕ?7.@	mfRTT??mfRTT??!mfRTT??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"ĕ?7.@QMI?????A?~???'@Y
e??k]??*	/?$?o@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??
}????!a^D֩F@)h>?n׻?1???GzZE@:Preprocessing2F
Iterator::Model?7k????!*??\Y?D@)?ԗ?????1??S<?+5@:Preprocessing2U
Iterator::Model::ParallelMapV2a8?0C???!]?|??3@)a8?0C???1]?|??3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??1>?^??!?;-?=(!@)?4Lkӈ?1~?_
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ??/-???!??U?9?@)Z??/-???1??U?9?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????!9??!?a??|M@)e?I)????1e??i@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorۈ'??q?!?b?Iu???)ۈ'??q?1?b?Iu???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]m???{??!???&aG@)f?O7P?m?1Kz?3????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??~j?tc?!??Ҡ???)??~j?tc?1??Ҡ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9mfRTT??I?u??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	QMI?????QMI?????!QMI?????      ??!       "      ??!       *      ??!       2	?~???'@?~???'@!?~???'@:      ??!       B      ??!       J	
e??k]??
e??k]??!
e??k]??R      ??!       Z	
e??k]??
e??k]??!
e??k]??b      ??!       JCPU_ONLYYmfRTT??b q?u??X@Y      Y@q?ӧ@(@"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?12.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 