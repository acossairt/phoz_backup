	?????}@?????}@!?????}@	???al	?????al	??!???al	??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????}@Qi??>???A?%W?x?}@Y??:??T??*	?S㥛?j@2U
Iterator::Model::ParallelMapV2???߃צ?!????4@)???߃צ?1????4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-z?m??!+?,9??8@)2q? ???1,?9??4@:Preprocessing2F
Iterator::Model~b????!??:??C@)C?????1ffn 3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateMi?-???!*?ƲC=@)?Z^??6??1k@???v1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?>?D???!?\8/'@)?>?D???1?\8/'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???UG???!?A?(N@)?74e???1?u?:J@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ne?΂?!B:??@)??ne?΂?1B:??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6?U?????!p*????>@)?q?j??l?1h??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9>??al	??IF?9i?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Qi??>???Qi??>???!Qi??>???      ??!       "      ??!       *      ??!       2	?%W?x?}@?%W?x?}@!?%W?x?}@:      ??!       B      ??!       J	??:??T????:??T??!??:??T??R      ??!       Z	??:??T????:??T??!??:??T??b      ??!       JCPU_ONLYY>??al	??b qF?9i?X@