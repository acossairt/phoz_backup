?	?_?-?`@?_?-?`@!?_?-?`@	??V???????V?????!??V?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?_?-?`@EIH?m???A_@/ܹ?`@Y!yv????*	n??ʹn@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN4?s???!	w??4U?@)z??L?D??1U-??r?9@:Preprocessing2U
Iterator::Model::ParallelMapV21??f???!?Ȃ???1@)1??f???1?Ȃ???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateur???7??!=Pj`P\;@)?a?[>???1!??@?-@:Preprocessing2F
Iterator::Model??e6??!=MSx@@)?ɐc???1??.z,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-Ӿ???!Y??`5)@)-Ӿ???1Y??`5)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?'?_[??!avY?C?P@)?????k??1|?߂?+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????!?&?x?@)??????1?&?x?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ڧ?1??!]|?|??>@)t(CUL??1?`9? %	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??V?????IG?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	EIH?m???EIH?m???!EIH?m???      ??!       "      ??!       *      ??!       2	_@/ܹ?`@_@/ܹ?`@!_@/ܹ?`@:      ??!       B      ??!       J	!yv????!yv????!!yv????R      ??!       Z	!yv????!yv????!!yv????b      ??!       JCPU_ONLYY??V?????b qG?????X@Y      Y@qD??EP@"?	
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?65.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 