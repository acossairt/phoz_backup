?	?-??i@?-??i@!?-??i@	??_???@??_???@!??_???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?-??i@?ۄ{e^??A[%X?h@Y???Wq!@*	t?V??@2F
Iterator::Model???@!?X5!
UU@)lC?83@1?:?v?tR@:Preprocessing2U
Iterator::Model::ParallelMapV2???O?s??!!?xS='@)???O?s??1!?xS='@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?^D?1u??!DB?N(@)kD0.??1C?d???$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceJӠh???!?>??W???)JӠh???1?>??W???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat됛?|??!s???wZ @)???[??1?"
N???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipE???JY??!?;U??W-@)MLb?G??1o?E??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorH?`?څ?!uWs??r??)H?`?څ?1uWs??r??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapū?m????!??4?p(@)<?$?t?1R$?lU??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??_???@I5*???W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ۄ{e^???ۄ{e^??!?ۄ{e^??      ??!       "      ??!       *      ??!       2	[%X?h@[%X?h@![%X?h@:      ??!       B      ??!       J	???Wq!@???Wq!@!???Wq!@R      ??!       Z	???Wq!@???Wq!@!???Wq!@b      ??!       JCPU_ONLYY??_???@b q5*???W@Y      Y@qN?oґ?"?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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