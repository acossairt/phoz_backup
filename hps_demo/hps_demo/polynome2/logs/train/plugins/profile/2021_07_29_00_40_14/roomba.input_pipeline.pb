	??K?AOS@??K?AOS@!??K?AOS@	?H???????H??????!?H??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??K?AOS@?Zd;?@AE?A?7R@Y???)???*	?x?&14p@2F
Iterator::ModelX9??v??!???L#?F@)??,????1?T~A8@:Preprocessing2U
Iterator::Model::ParallelMapV2??p<???!e?_?5@)??p<???1e?_?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?fHū??!AUf*9@)jL???j??1??6m?4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatek???T??!%Yԧ??6@)?-v??2??1w?????,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??0??B??!??? @)??0??B??1??? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor`"ĕ???!?8н??@)`"ĕ???1?8н??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?o????!5&??K@)]7??VB??1??١??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{?%T??!???%?8@)c????r?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?H??????I\????X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Zd;?@?Zd;?@!?Zd;?@      ??!       "      ??!       *      ??!       2	E?A?7R@E?A?7R@!E?A?7R@:      ??!       B      ??!       J	???)??????)???!???)???R      ??!       Z	???)??????)???!???)???b      ??!       JCPU_ONLYY?H??????b q\????X@