	???o?c@???o?c@!???o?c@	?Nx?rθ??Nx?rθ?!?Nx?rθ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???o?c@n??)XX@A'??rJ?N@Y#?Ƥ???*	~?5^?eh@2F
Iterator::ModelU???????!N? ?I@)?8K?r??1?2,?jw:@:Preprocessing2U
Iterator::Model::ParallelMapV2?-$`t??!~i?)?x9@)?-$`t??1~i?)?x9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
J?ʽ???!ai?`?4@)?&?5???1??????0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?A
?B???!X?????4@)%??????1??)2??+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???I??!PB?1dN@)???I??1PB?1dN@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?'+????!??n??H@)?\??X3??1??͈6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;??Tގ??!??f?đ@);??Tގ??1??f?đ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[??ù??!̊????6@)?e6\p?1?#??^ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Nx?rθ?I??Bc??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	n??)XX@n??)XX@!n??)XX@      ??!       "      ??!       *      ??!       2	'??rJ?N@'??rJ?N@!'??rJ?N@:      ??!       B      ??!       J	#?Ƥ???#?Ƥ???!#?Ƥ???R      ??!       Z	#?Ƥ???#?Ƥ???!#?Ƥ???b      ??!       JCPU_ONLYY?Nx?rθ?b q??Bc??X@