?	>x?҆?M@>x?҆?M@!>x?҆?M@	??Ϻ7$????Ϻ7$??!??Ϻ7$??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$>x?҆?M@?nI????A>?x??@M@Y=D?;????*	n???Ot@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??A?p???!?FJ]?E@)W@??>??1??+G???@:Preprocessing2F
Iterator::Model????:8??!?p??@@)V???4??1?dA[??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat*?T???!-?O??1@)??&????1?;a???,@:Preprocessing2U
Iterator::Model::ParallelMapV2l?6???!??mky*@)l?6???1??mky*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???$??!4???H'@)???$??14???H'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W}??!????+?P@)?z?V????1/???K?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn½2oՅ?!#S?V>
@)n½2oՅ?1#S?V>
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!Cu?*?bF@)	?Į??v?1ߔ|???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??Ϻ7$??I
0E???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?nI?????nI????!?nI????      ??!       "      ??!       *      ??!       2	>?x??@M@>?x??@M@!>?x??@M@:      ??!       B      ??!       J	=D?;????=D?;????!=D?;????R      ??!       Z	=D?;????=D?;????!=D?;????b      ??!       JCPU_ONLYY??Ϻ7$??b q
0E???X@Y      Y@qt.??''N@"?	
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?60.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 