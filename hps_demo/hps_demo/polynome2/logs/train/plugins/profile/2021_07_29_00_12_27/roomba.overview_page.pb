?	6>?????@6>?????@!6>?????@	|???????|???????!|???????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$6>?????@???"???A???0ln?@Yޮ??p@*	B`??r??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateU?t<f?&@!?g`???T@)?q4GV?&@1?Z?Z+?T@:Preprocessing2U
Iterator::Model::ParallelMapV2I??_?`??!??_??? @)I??_?`??1??_??? @:Preprocessing2F
Iterator::Model遏??? @!4K:?/@)G??1????Of@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??S^??!H??/????)V*?????1.???bk??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice]?@???!_?(???)]?@???1_?(???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipc??Ր'@!??8O?U@)k:!tЕ?1???Y????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?D?A???!kXP??e??)?D?A???1kXP??e??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaput\???&@!F\???T@)??;?2t?1????y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9}???????Ip5r?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???"??????"???!???"???      ??!       "      ??!       *      ??!       2	???0ln?@???0ln?@!???0ln?@:      ??!       B      ??!       J	ޮ??p@ޮ??p@!ޮ??p@R      ??!       Z	ޮ??p@ޮ??p@!ޮ??p@b      ??!       JCPU_ONLYY}???????b qp5r?X@Y      Y@q5?n?Y($@"?	
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?10.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 