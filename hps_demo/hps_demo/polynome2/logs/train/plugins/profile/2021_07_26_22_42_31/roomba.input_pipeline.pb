	\v???M@\v???M@!\v???M@	H?2Y?s??H?2Y?s??!H?2Y?s??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\v???M@??????3@Ar?	?O?C@Y?L??????*	?x?&~?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??_Yib@!(m}EX@)?O?mP@1?FB
P-X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?H?H???!4??E??)/2?F???1?Pu?D8??:Preprocessing2F
Iterator::Model???u6???!z?VE??)?&"???15??????:Preprocessing2U
Iterator::Model::ParallelMapV2`W??????!?ID?@??)`W??????1?ID?@??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????O??!??&?-??)????O??1??&?-??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???@!???ʫX@)? ?w?~??1?u???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Y?N܃?!Y?J?8??)?Y?N܃?1Y?J?8??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?>?̔f@!Vtb?JX@)??D-ͭp?1޺?~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 33.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9H?2Y?s??IߚM??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????3@??????3@!??????3@      ??!       "      ??!       *      ??!       2	r?	?O?C@r?	?O?C@!r?	?O?C@:      ??!       B      ??!       J	?L???????L??????!?L??????R      ??!       Z	?L???????L??????!?L??????b      ??!       JCPU_ONLYYH?2Y?s??b qߚM??X@