	ԁ??VH@ԁ??VH@!ԁ??VH@	??FY?????FY???!??FY???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ԁ??VH@???1Z7@A?EaE9D@Y3??J&??*	0?$??i@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU1?~±?!?????@@)??V????1oiX??;@:Preprocessing2F
Iterator::Model?K⬈???!??J??xC@)?,?i????1??C?q5@:Preprocessing2U
Iterator::Model::ParallelMapV2?j???u??!?R?q1@)?j???u??1?R?q1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatex|{נ/??!?M?E?4@)?-???1??1?G'???$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?fI-??!kSs?a#@)?fI-??1kSs?a#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/?HM??!a? ?@)/?HM??1a? ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip|??&??!$4?`?N@)?!?{???1?H???F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\??.?u??!]?mw?+6@)W#??2r?1????2@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??FY???I[??/?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???1Z7@???1Z7@!???1Z7@      ??!       "      ??!       *      ??!       2	?EaE9D@?EaE9D@!?EaE9D@:      ??!       B      ??!       J	3??J&??3??J&??!3??J&??R      ??!       Z	3??J&??3??J&??!3??J&??b      ??!       JCPU_ONLYY??FY???b q[??/?X@